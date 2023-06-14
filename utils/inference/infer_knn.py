# Copyright 2023 - Valeo Comfort and Driving Assistance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Inference with the KNN post-processing step (in this case the KPConv layer is not used).

import torch
import argparse
import numpy as np
import time
import datetime
import os
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../')
sys.path.append('../rangevit')

from option import Option
from main import build_rangevit_model
from rangevit import dataset
from rangevit import models
from rangevit import utils
from rangevit.utils.tools import Recorder
from rangevit.utils.metrics.eval_results import eval_results
from inference_utils import inference


class Inference(object):
    def __init__(self, settings: Option, model: nn.Module, recorder: Recorder):
        self.settings = settings
        self.model = model.cuda()
        self.recorder = recorder

        self.use_knn = settings.use_knn

        knn_params = {
            'knn': 5,
            'search': settings.knn_search,
            'sigma': 1.0,
            'cutoff': 1.0,
        }

        self.knn_post = utils.postproc.KNN(
            params=knn_params,
            nclasses=self.settings.n_classes)

        if self.use_knn and self.recorder is not None:
            self.recorder.logger.info('Using KNN Post Process')

        self.val_loader, self.val_range_loader = self._initDataloader()

        self.prediction_path = os.path.join(self.settings.save_path, 'preds')

        self.evaluator = utils.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device('cpu'),
            ignore=[0], is_distributed=self.settings.distributed)
        
        self.pixel_eval = utils.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device('cpu'),
            ignore=[0], is_distributed=self.settings.distributed)

        if self.settings.has_label:
            self.data_split = 'val'
        else:
            self.data_split = 'test'

    def _initDataloader(self):
        if self.settings.dataset.lower() in 'nuscenes':
            if self.settings.has_label:
                version = 'v1.0-mini' if self.settings.use_mini_version else 'v1.0-trainval'
                split='val'
            else:
                version = 'v1.0-test'
                split = 'test'

            valset = dataset.nuScenes.Nuscenes(
                root=self.settings.data_root, version=version, split=split, has_image=False)

        elif self.settings.dataset.lower() == 'semantickitti':
            valset = dataset.semantic_kitti.SemanticKitti(
                root=self.settings.data_root,
                sequences=[8],
                config_path='dataset/semantic_kitti/semantic-kitti.yaml',
                has_label=self.settings.has_label,
                has_image=False)
        else:
            raise ValueError('invalid dataset: {}'.format(self.settings.dataset))

        val_range_loader = dataset.RangeViewLoader(
                dataset=valset,
                config=self.settings.config,
                is_train=False, 
                return_uproj=True, 
                use_kpconv=False)

        sampler = torch.utils.data.DistributedSampler(val_range_loader, shuffle=False)
        val_loader = torch.utils.data.DataLoader(
            val_range_loader,
            batch_size=1,
            sampler=sampler,
            num_workers=self.settings.num_workers,
            shuffle=False,
            drop_last=False)

        return val_loader, val_range_loader

    def run(self):
        self.model.eval()
        self.evaluator.reset()
        self.pixel_eval.reset()

        model_without_ddp = self.model
        if hasattr(self.model, 'module'):
            model_without_ddp = self.model.module
        
        with torch.no_grad():
            t_start = time.time()
            for i, (input_feature, input_label, _, proj_depth, uproj_x_idx, uproj_y_idx, uproj_depth, sem_label) in enumerate(self.val_loader):
                t_process_start = time.time()             
                
                # Feature: range, x, y, z, intensity
                input_feature = input_feature.cuda() # shape: 1 x 5 x H x W

                input_label = input_label.long().cuda()
                uproj_x_idx = uproj_x_idx[0].cuda()
                uproj_y_idx = uproj_y_idx[0].cuda()
                uproj_depth = uproj_depth[0].cuda()
                proj_depth = proj_depth[0].cuda()
                
                assert input_feature.shape[0] == 1 # batch size has to be 1

                # Inference
                im_meta = dict(flip=False)
                pred_output = inference(
                    model_without_ddp.rangevit,
                    [input_feature],
                    [im_meta],
                    ori_shape=input_feature.shape[2:4],
                    window_size=self.settings.window_size,
                    window_stride=self.settings.window_stride,
                    batch_size=1,
                    use_kpconv=False)
                
                pred_output = pred_output.unsqueeze(0)  # shape: 1 x n_cls x H x W
                pred_output = F.softmax(pred_output, dim=1)
                pred_argmax = pred_output[0].argmax(dim=0)
                argmax = pred_output.argmax(dim=1)

                t_model_process_end = time.time()
                model_process_time = t_model_process_end - t_process_start
                
                # KNN post process
                if self.use_knn:
                    unproj_argmax = self.knn_post(
                        proj_depth,
                        uproj_depth,
                        pred_argmax,
                        uproj_x_idx,
                        uproj_y_idx)
                else:
                    unproj_argmax = pred_argmax[uproj_y_idx, uproj_x_idx]

                post_process_time = time.time() - t_model_process_end

                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)

                # Measure IoU
                self.evaluator.addBatch(pred_np, sem_label) # 3D predictions
                self.pixel_eval.addBatch(argmax, input_label) # 2D predictions

                # Save the predictions
                if self.settings.save_eval_results:
                    if self.settings.dataset == 'NuScenes':
                        pred_path = os.path.join(self.prediction_path, 'lidarseg', self.data_split)
                        lidar_token = self.val_range_loader.dataset.token_list[i]
                        
                        if not os.path.isdir(pred_path):
                            os.makedirs(pred_path)
                        pred_result_path = os.path.join(pred_path, '{}_lidarseg.bin'.format(lidar_token))
                        pred_np.tofile(pred_result_path)

                    elif self.settings.dataset == 'SemanticKitti':
                        pred_np_origin = self.val_range_loader.dataset.class_map_lut_inv[pred_np]
                        seq_id, frame_id = self.val_range_loader.dataset.parsePathInfoByIndex(i)
                        pred_path = os.path.join(self.prediction_path, 'sequences', seq_id, 'predictions')
                        
                        if not os.path.isdir(pred_path):
                            os.makedirs(pred_path)
                        pred_result_path = os.path.join(pred_path, '{}.label'.format(frame_id))
                        pred_np_origin.tofile(pred_result_path)

                t_proces_end = time.time()
                process_time = t_proces_end - t_process_start
                data_time = t_process_start - t_start
                t_start = time.time()

                if (i % self.settings.log_frequency) == 0 or (i+1) == len(self.val_loader):
                    print('Iter [{:04d}|{:04d}] Datatime: {:0.3f} ProcessTime: {:0.3f} with ModelProcessTime: {:0.3f} and PostProcessTime: {:0.3f}'.format(
                        i, len(self.val_loader), data_time, process_time, model_process_time, post_process_time))


        if not self.settings.has_label:
            return


        with torch.no_grad():
            # 3D results
            mean_acc, class_acc = self.evaluator.getAcc()
            mean_recall, class_recall = self.evaluator.getRecall()
            mean_iou, class_iou = self.evaluator.getIoU()
        
            metrics_dict_3d = {
                'mean_acc': mean_acc,
                'class_acc': class_acc,
                'mean_recall': mean_recall,
                'class_recall': class_recall,
                'mean_iou': mean_iou,
                'class_iou': class_iou,
                'conf_matrix': self.evaluator.conf_matrix.clone().cpu(),
            }

            # 2D results
            mean_acc, class_acc = self.pixel_eval.getAcc()
            mean_recall, class_recall = self.pixel_eval.getRecall()
            mean_iou, class_iou = self.pixel_eval.getIoU()
        
            metrics_dict_2d = {
                'mean_acc': mean_acc,
                'class_acc': class_acc,
                'mean_recall': mean_recall,
                'class_recall': class_recall,
                'mean_iou': mean_iou,
                'class_iou': class_iou,
                'conf_matrix': self.pixel_eval.conf_matrix.clone().cpu(),
            }

        if self.recorder is not None:
            # Print point-wise results (3D eval)
            eval_results(pixel_or_point='Point',
                         settings=self.settings,
                         recorder=self.recorder,
                         metrics_dict=metrics_dict_3d,
                         dataloader=self.val_range_loader,
                         print_data_distribution=True)

            # Print pixel-wise results (2D eval)
            eval_results(pixel_or_point='Pixel',
                         settings=self.settings,
                         recorder=self.recorder,
                         metrics_dict=metrics_dict_2d,
                         dataloader=self.val_range_loader,
                         print_data_distribution=True)


class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings

        # Init gpu
        utils.tools.init_distributed_mode(self.settings)
        torch.distributed.barrier()

        self.settings.check_path()

        # Set random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        torch.cuda.set_device(self.settings.gpu)
        torch.backends.cudnn.benchmark = True

        self.recorder = None
        if not self.settings.distributed or (self.settings.rank == 0):
            self.recorder = utils.tools.Recorder(
                self.settings, self.settings.save_path, use_tensorboard=False)

        # Init model
        self.model = build_rangevit_model(self.settings, 
                                          self.settings.pretrained_model)

        # Load checkpoint
        self._loadCheckpoint()

        # Init inference
        self.inference = Inference(self.settings, self.model, self.recorder)


    def _loadCheckpoint(self):
        if self.settings.pretrained_model is not None:
            if not os.path.isfile(self.settings.pretrained_model):
                raise FileNotFoundError('pretrained model not found: {}'.format(
                    self.settings.pretrained_model))

            pretrained_dict = torch.load(self.settings.pretrained_model, map_location='cpu')

            if 'model' in pretrained_dict.keys():
                self.model.load_state_dict(pretrained_dict['model'])
            else:
                self.model.load_state_dict(pretrained_dict)

            if self.recorder is not None:
                self.recorder.logger.info(
                    'loading pretrained weight from: {}'.format(self.settings.pretrained_model))

    def run(self):
        t_start = time.time()
        self.inference.run()
        cost_time = time.time() - t_start
        self.recorder.logger.info('==== Total cost time: {}'.format(
            datetime.timedelta(seconds=cost_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment Options')
    parser.add_argument('config_path', type=str, metavar='config_path',
                        help='path of config file, type: string')
    parser.add_argument('--data_root', type=str,
                        help='path to the data, type: string')
    parser.add_argument('--save_path', type=str,
                        help='path to save the file, type: string')
    parser.add_argument('--id', type=str, 
                        help='name to identify the run')

    parser.add_argument('--pretrained_model', type=str, 
                        help='path of pre-trained model, type: string')
    parser.add_argument('--mini', type=bool, help='use mini version of the dataset, type: bool')
    parser.add_argument('--save_eval_results', type=bool)
    parser.add_argument('--log_frequency', type=int)
    parser.add_argument('--knn', type=bool, default=False)
    parser.add_argument('--knn_search', type=int, default=13)

    args = parser.parse_args()
    settings = Option(args.config_path, args)

    settings.use_kpconv = False # KPConv is not used with this inference code

    settings.data_root = args.data_root if args.data_root is not None else settings.data_root
    settings.save_path = args.save_path if args.save_path is not None else settings.save_path
    settings.id = args.id if args.id is not None else settings.id
    settings.pretrained_model = args.pretrained_model if args.pretrained_model is not None else settings.pretrained_model
    settings.use_mini_version = args.mini if args.mini is not None else settings.use_mini_version
    settings.save_eval_results = args.save_eval_results if args.save_eval_results is not None else settings.save_eval_results
    settings.log_frequency = args.log_frequency if args.log_frequency is not None else settings.log_frequency

    settings.use_knn = args.knn
    settings.knn_search = args.knn_search

    if not os.path.isdir(settings.save_path):
        raise ValueError('Training path does not exists: {}'.format(settings.save_path))

    knn_str = f'_KNN_{args.knn_search}' if args.knn else ''
    settings.save_path = os.path.join(
        settings.save_path, f'Eval{knn_str}_{args.id}')

    exp = Experiment(settings)
    exp.run()
