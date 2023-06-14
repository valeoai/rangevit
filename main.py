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

import torch
import argparse
import os
import datetime
import time
import numpy as np

from option import Option
from train import Trainer
import models
import utils
import utils.tools as tools
from models.model_utils import resize_pos_embed


def build_rangevit_model(settings, pretrained_path=None):
    model = models.RangeViT(
        in_channels=settings.in_channels,
        n_cls=settings.n_classes,
        backbone=settings.vit_backbone,
        image_size=settings.image_size,
        pretrained_path=pretrained_path,
        new_patch_size=settings.patch_size,
        new_patch_stride=settings.patch_stride,
        reuse_pos_emb=settings.reuse_pos_emb,
        reuse_patch_emb=settings.reuse_patch_emb,
        conv_stem=settings.conv_stem,
        stem_base_channels=settings.stem_base_channels,
        stem_hidden_dim=settings.D_h,
        skip_filters=settings.skip_filters,
        decoder=settings.decoder,
        up_conv_d_decoder=settings.D_h,
        up_conv_scale_factor=settings.patch_stride,
        use_kpconv=settings.use_kpconv)
    return model


class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings

        # Init gpu
        tools.init_distributed_mode(self.settings)
        torch.distributed.barrier()

        self.settings.check_path()

        # Set random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        np.random.seed(self.settings.seed)
        torch.cuda.set_device(self.settings.gpu)
        torch.backends.cudnn.benchmark = True

        # Init checkpoint
        self.recorder = None
        if tools.is_main_process():
            self.recorder = utils.tools.Recorder(self.settings, self.settings.save_path)

        self.prediction_path = os.path.join(self.settings.save_path, 'preds')

        self.epoch_start = 0

        # Init model
        self.model = self._initModel()

        # Init trainer
        self.trainer = Trainer(self.settings, self.model, self.recorder)

        # Load checkpoint
        self._loadCheckpoint()


    def _initModel(self):
        # Model
        model = build_rangevit_model(
            self.settings,
            pretrained_path=self.settings.pretrained_model)

        # Freezing the ViT encoder weights.
        if self.settings.freeze_vit_encoder:
            print('==> Freeze the ViT encoder (without the pos_embed and stem)')
            for param in model.rangevit.encoder.blocks.parameters():
                param.requires_grad = False

            model.rangevit.encoder.norm.weight.requires_grad = False
            model.rangevit.encoder.norm.bias.requires_grad = False

            # Unfreeze the LayerNorm layers
            if self.settings.unfreeze_layernorm:
                print('==> Unfreeze the LN layers')
                model.rangevit.encoder.norm.weight.requires_grad = True
                model.rangevit.encoder.norm.bias.requires_grad = True
                for block_id in range(0, len(model.rangevit.encoder.blocks)):
                    model.rangevit.encoder.blocks[block_id].norm1.weight.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].norm1.bias.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].norm2.weight.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].norm2.bias.requires_grad = True

            if self.settings.unfreeze_attn:
                print('==> Unfreeze the ATTN layers: qkv and proj')
                for block_id in range(0, len(model.rangevit.encoder.blocks)):
                    model.rangevit.encoder.blocks[block_id].attn.qkv.weight.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].attn.qkv.bias.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].attn.proj.weight.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].attn.proj.bias.requires_grad = True

            if self.settings.unfreeze_ffn:
                print('==> Unfreeze the FFN layers: mlp.fc1 and mlp.fc2')
                for block_id in range(0, len(model.rangevit.encoder.blocks)):
                    model.rangevit.encoder.blocks[block_id].mlp.fc1.weight.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].mlp.fc1.bias.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].mlp.fc2.weight.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].mlp.fc2.bias.requires_grad = True


        if self.recorder is not None:
            self.recorder.logger.info(f'model = {model}')

            stats = model.counter_model_parameters()
            if hasattr(model, 'counter_model_parameters'):
                self.recorder.logger.info(f'Number of model parameters:')
                for key, val in stats.items():
                    self.recorder.logger.info(f'==> {key}: {val}')

        return model


    def _loadCheckpoint(self):
        if self.settings.checkpoint is not None:
            print(f'Resume training from checkpoint {self.settings.checkpoint}')
            if not os.path.isfile(self.settings.checkpoint):
                raise FileNotFoundError('checkpoint file not found: {}'.format(self.settings.checkpoint))

            checkpoint_data = torch.load(self.settings.checkpoint, map_location='cpu')

            if self.settings.finetune_pretrained_model:
                # When fine-tuning a segmentation model previously pre-trained to another dataset then it
                # is necessary to adapt the (a) pos_embeds and (b) to remove the classification head.
                image_size = self.model.rangevit.encoder.image_size
                patch_stride = self.model.rangevit.encoder.patch_stride
                if (self.model.rangevit.encoder.pos_embed.shape != checkpoint_data['model']['rangevit.encoder.pos_embed'].shape):
                    assert self.model.rangevit.encoder.pos_embed.shape[2] == checkpoint_data['model']['rangevit.encoder.pos_embed'].shape[2]
                    gs_new_h = int(image_size[0] // patch_stride[0])
                    gs_new_w = int(image_size[1] // patch_stride[1])
                    num_extra_tokens = 1
                    assert (gs_new_h * gs_new_w + num_extra_tokens) == self.model.rangevit.encoder.pos_embed.shape[1]
                    old_len = checkpoint_data['model']['rangevit.encoder.pos_embed'].shape[1] - num_extra_tokens # remove one for the classification token

                    gs_old_w = gs_new_w
                    gs_old_h = old_len // gs_old_w
                    checkpoint_data['model']['rangevit.encoder.pos_embed'] = (
                        resize_pos_embed(checkpoint_data['model']['rangevit.encoder.pos_embed'],
                                         grid_old_shape=(gs_old_h, gs_old_w),
                                         grid_new_shape=(gs_new_h, gs_new_w),
                                         num_extra_tokens=num_extra_tokens))
                assert self.model.rangevit.encoder.pos_embed.shape == checkpoint_data['model']['rangevit.encoder.pos_embed'].shape

                for key in ('rangevit.kpclassifier.head.weight', 'rangevit.kpclassifier.head.bias'):
                    del checkpoint_data['model'][key]

            checkpoint_data_model = checkpoint_data['model']
            msg = self.model.load_state_dict(checkpoint_data_model, strict=(not self.settings.finetune_pretrained_model))
            #print(f'msg = {msg}')

            if not self.settings.finetune_pretrained_model:
                print(f'==> Loading optimizer')
                if self.settings.val_only is False:
                    self.trainer.optimizer.load_state_dict(checkpoint_data['optimizer'])
                self.epoch_start = checkpoint_data['epoch'] + 1

                if ('fp16_scaler' in checkpoint_data) and (checkpoint_data['fp16_scaler'] is not None):
                    self.trainer.fp16_scaler.load_state_dict(checkpoint_data['fp16_scaler'])


    def run(self):
        t_start = time.time()
        if self.settings.val_only:
            save_results_path = self.prediction_path if self.settings.save_eval_results else None
            self.trainer.run(self.epoch_start,
                             mode='Validation',
                             print_results=True,
                             save_results_path=save_results_path)

            cost_time = time.time() - t_start
            if self.recorder is not None:
                self.recorder.logger.info('==== Total cost time: {}'.format(
                    datetime.timedelta(seconds=cost_time)))
            return
        best_val_result = None

        #self.trainer.scheduler.step(self.epoch_start*len(self.trainer.train_loader))

        for epoch in range(self.epoch_start, self.settings.n_epochs):

            # Run one epoch
            self.trainer.run(epoch, mode='Train')

            # Run validation
            if (epoch % self.settings.val_frequency == 0 or
                epoch == self.settings.n_epochs - 1 or
                epoch == self.epoch_start):
                val_result = self.trainer.run(epoch, mode='Validation')

                # Save the best result
                if self.recorder is not None:
                    self.recorder.logger.info(f'---- Best result after Epoch {epoch+1} ----')
                    if best_val_result is None:
                        best_val_result = val_result
                    for k, v in val_result.items():
                        if v >= best_val_result[k]:
                            self.recorder.logger.info(
                                'Get better {} model: {}'.format(k, v))
                            saved_path = os.path.join(
                                self.recorder.checkpoint_path, 'best_{}_model.pth'.format(k))
                            saved_path_start = os.path.join(
                                self.recorder.checkpoint_path, 'best_{}_model_from_start_{}.pth'.format(k, self.epoch_start))
                            best_val_result[k] = v

                            checkpoint_data = {
                                'model': self.model.state_dict(),
                                'optimizer': self.trainer.optimizer.state_dict(),
                                'epoch': epoch,
                                k: v,
                            }

                            if self.trainer.fp16_scaler is not None:
                                checkpoint_data['fp16_scaler'] = self.trainer.fp16_scaler.state_dict()

                            torch.save(checkpoint_data, saved_path)
                            if self.epoch_start > 0:
                                torch.save(checkpoint_data, saved_path_start)

            # Save checkpoint
            if self.recorder is not None:
                saved_path = os.path.join(self.recorder.checkpoint_path, 'checkpoint.pth')

                checkpoint_data = {
                    'model': self.model.state_dict(),
                    'optimizer': self.trainer.optimizer.state_dict(),
                    'epoch': epoch,
                }
                if self.trainer.fp16_scaler is not None:
                    checkpoint_data['fp16_scaler'] = self.trainer.fp16_scaler.state_dict()
                torch.save(checkpoint_data, saved_path)

                # Logging best results
                if best_val_result is not None:
                    log_str = '>>> Best Result: '
                    for k, v in best_val_result.items():
                        log_str += '{}: {} '.format(k, v)
                    log_str += '\n'
                    self.recorder.logger.info(log_str)

        # Print total cost time
        cost_time = time.time() - t_start
        if self.recorder is not None:
            self.recorder.logger.info('=== Total cost time: {}'.format(
                datetime.timedelta(seconds=cost_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment Options')
    parser.add_argument('config_path', type=str, metavar='config_path',
                        help='path of config file, type: string')
    parser.add_argument('--data_root', type=str, required=True,
                        help='path to the data, type: string')
    parser.add_argument('--save_path', type=str, required=True,
                        help='path to save the file, type: string')
    parser.add_argument('--id', type=str,
                        help='name to identify the run')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of threads used for data loading, type: int')
    parser.add_argument('--pretrained_model', type=str,
                        help='path of pre-trained model to initialize the ViT encoder backbone, type: string')
    parser.add_argument('--checkpoint', type=str,
                        help='path of checkpoint model for resuming training or evaluation, type: string')
    parser.add_argument('--window_stride', type=int,
                        help='sliding window stride during validation, type: int')
    parser.add_argument('--mini', action='store_true', help='use mini version of the dataset, type: bool')
    parser.add_argument('--val_only', action='store_true', help='run inference only')
    parser.add_argument('--test_split', action='store_true', help='run inference on the test split')
    parser.add_argument('--save_eval_results', action='store_true', help='save the predictions')
    parser.add_argument('--log_frequency', type=int, default=100, help='logging frequency')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    settings = Option(args.config_path, args)

    settings.id = args.id if args.id is not None else settings.id
    settings.pretrained_model = args.pretrained_model if args.pretrained_model is not None else settings.pretrained_model

    if args.checkpoint is not None:
        settings.checkpoint = args.checkpoint
        settings.pretrained_model = None
        settings.finetune_pretrained_model = False

    if args.val_only and args.window_stride is not None:
        settings.window_stride = [settings.window_stride[0], args.window_stride]
        print(f'WINDOW STRIDE: {settings.window_stride}')

    settings.data_root = args.data_root
    settings.use_mini_version = args.mini
    settings.val_only = args.val_only
    settings.test_split = args.test_split
    settings.save_eval_results = args.save_eval_results
    settings.log_frequency = args.log_frequency
    settings.num_workers = args.num_workers
    settings.seed = args.seed

    # No patch and positional embeddings are loaded when training from scratch.
    if settings.pretrained_model is None:
        settings.reuse_patch_emb = False
        settings.reuse_pos_emb = False

    if settings.val_only:
        settings.save_path = os.path.join(settings.save_path, f'Eval_{settings.id}')

    exp = Experiment(settings)
    exp.run()
