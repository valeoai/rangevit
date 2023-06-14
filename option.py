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

import os
import yaml
import sys
import shutil
import utils.tools as tools


class Option(object):
    def __init__(self, config_path, args):
        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, 'r'))

        # General options
        self.seed = 1
        self.gpu = None
        self.rank = 0  # rank of distributed thread
        self.world_size = 1
        self.distributed = False
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        self.num_workers = 4 # number of threads used for data loading

        # Data config
        self.dataset = self.config['dataset']
        self.n_classes = self.config['n_classes']
        self.data_root = None
        self.has_label = self.config['has_label']
        self.use_mini_version = False
        self.use_trainval = self.config.get('use_trainval', False)

        # Train config
        self.val_only = False
        self.val_frequency = self.config.get('val_frequency', 10)
        self.test_split = False
        self.n_epochs = self.config['n_epochs']  # number of total epochs
        self.batch_size = self.config['batch_size']  # mini-batch size
        self.batch_size_val = self.config.get('batch_size_val', 1) # validation batch size
        self.lr = self.config['lr']
        self.warmup_epochs = self.config.get('warmup_epochs', 10)
        self.log_frequency = 100
        self.train_result_frequency = self.config.get('train_result_frequency', 100)
        self.use_fp16 = self.config.get('use_fp16', False) # for mixed-precision training


        # Model config
        self.vit_backbone = self.config.get('vit_backbone', 'vit_small_patch16_384')
        self.in_channels = self.config.get('in_channels', 5)
        self.patch_size = self.config.get('patch_size', [2, 8])
        self.patch_stride = self.config.get('patch_stride', [2, 8])
        self.image_size = self.config.get('image_size', [32, 384])
        self.window_size = self.config.get('window_size', [32, 384])
        self.window_stride = self.config.get('window_stride', [32, 256])
        self.original_image_size = self.config.get('original_image_size', [32, 2048])

        # Freeze encoder params
        self.freeze_vit_encoder = self.config.get('freeze_vit_encoder', False)
        self.unfreeze_layernorm = self.config.get('unfreeze_layernorm', False)
        self.unfreeze_attn = self.config.get('unfreeze_attn', False)
        self.unfreeze_ffn = self.config.get('unfreeze_ffn', False)

        # Stem
        self.conv_stem = self.config.get('conv_stem', 'ConvStem')
        self.stem_base_channels = self.config.get('stem_base_channels', 32)
        self.D_h = self.config.get('D_h', 256)

        # Decoder
        self.decoder = self.config.get('decoder', 'up_conv')
        self.skip_filters = self.config.get('skip_filters', 0)

        # 3D refiner
        self.use_kpconv = self.config.get('use_kpconv', True)


        # Checkpoint model
        self.checkpoint = self.config.get('checkpoint', None)
        self.pretrained_model = self.config.get('pretrained_model', None)
        self.finetune_pretrained_model = self.config.get('finetune_pretrained_model', False)

        # Loading pre-trained patch and positional embeddings
        self.reuse_pos_emb = self.config.get('reuse_pos_emb', False)
        self.reuse_patch_emb = self.config.get('reuse_patch_emb', False)


        # Save results
        self.id = self.config['id'] # name to identify the run
        self.save_eval_results = False

        self.save_path = args.save_path
        self.save_path = os.path.join(self.save_path, 'log_{}'.format(self.id))


        # -----------------------------------------------------
        # Check options

        # There is no skip connection if no convolutional stem is used or the linear decoder is used.
        # (If no convolutional stem is used, then we use PatchEmbedding istead).
        if self.conv_stem == 'none' or self.decoder == 'linear':
            assert self.skip_filters == 0

        # If there is a skip connection, it's channel dim has to be D_h.
        if self.skip_filters > 0:
            assert self.skip_filters == self.D_h

        # If a convolutional stem is used, patch_size = patch_stride and there is no patch embedding
        # so we can't load pre-trained weights in the patch embeddings.
        if self.conv_stem != 'none':
            assert self.patch_size == self.patch_stride
            assert self.reuse_patch_emb == False

        # When using the KPConv layer, the decoder has to be up_conv.
        if self.use_kpconv:
            assert self.decoder == 'up_conv'

        # The following hyperparameters have to be tuples or lists with two elements.
        tuple_list = [self.patch_size, self.patch_stride,
                      self.image_size, self.window_size, self.window_stride,
                      self.original_image_size]
        for i in tuple_list:
            assert isinstance(i, (list, tuple))
            assert len(i) == 2

        # No patch and positional embeddings are loaded when training from scratch.
        if self.pretrained_model == None:
            assert self.reuse_patch_emb == self.reuse_pos_emb == False


    def check_path(self):
        if tools.is_main_process():
            if os.path.exists(self.save_path):
                print('WARNING: Directory exist: {}'.format(self.save_path))

            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
