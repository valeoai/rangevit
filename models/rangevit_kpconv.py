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
import os
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import padding, unpadding
from .kpconv.blocks import KPConv


def resample_grid(predictions, py, px):
    pypx = torch.stack([px, py], dim=3)
    resampled = F.grid_sample(predictions, pypx)
    return resampled


class KPClassifier(nn.Module):
    # Adapted from D. Kochanov et al.
    # https://github.com/DeyvidKochanov-TomTom/kprnet
    def __init__(self, in_channels=256, out_channels=256, num_classes=17, dummy=False):
        super(KPClassifier, self).__init__()
        self.kpconv = KPConv(
            kernel_size=15,
            p_dim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            KP_extent=1.2,
            radius=0.60,
        )
        self.dummy = dummy
        if self.dummy:
            del self.kpconv
            self.kpconv = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.head = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, x, px, py, pxyz, pknn, num_points):
        assert px.shape[0] == py.shape[0]
        assert px.shape[0] == pxyz.shape[0]
        assert px.shape[0] == pknn.shape[0]
        assert px.shape[0] == num_points.sum().item()
        res = []
        offset = 0
        batch_size = x.shape[0]
        for i in range(batch_size):
            len = num_points[i]
            px_i = px[offset:(offset+len)].unsqueeze(0).unsqueeze(1).contiguous()
            py_i = py[offset:(offset+len)].unsqueeze(0).unsqueeze(1).contiguous()
            points = pxyz[offset:(offset+len)].contiguous()
            pknn_i = pknn[offset:(offset+len)].contiguous()
            resampled = F.grid_sample(
                x[i].unsqueeze(0), torch.stack([px_i, py_i], dim=3),
                align_corners=False, padding_mode='border')
            feats = resampled.squeeze().t()

            if feats.shape[0] != points.shape[0]:
                print(f'feats.shape={feats.shape} vs points.shape={points.shape}')
            assert feats.shape[0] == points.shape[0]
            if self.dummy:
                feats = self.kpconv(feats)
            else:
                feats = self.kpconv(points, points, pknn_i, feats)
            res.append(feats)
            offset += len

        assert offset == px.shape[0]
        res = torch.cat(res, axis=0).unsqueeze(2).unsqueeze(3)
        res = self.relu(self.bn(res))
        return self.head(res)


class RangeViT_KPConv(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        kpclassifier,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.patch_stride = encoder.patch_stride
        self.encoder = encoder
        self.decoder = decoder
        del self.decoder.head
        self.kpclassifier = kpclassifier

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay('encoder.', self.encoder).union(
            append_prefix_no_weight_decay('decoder.', self.decoder)
        )
        return nwd_params

    def forward_2d_features(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        
        x, skip = self.encoder(im, return_features=True) # x.shape = [16, 577, 384]
        
        # remove CLS tokens for decoding
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:] # x.shape = [16, 576, 384]
        
        feats = self.decoder(x, (H, W), skip, return_features=True)
        feats = F.interpolate(feats, size=(H, W), mode='bilinear', align_corners=False)
        feats = unpadding(feats, (H_ori, W_ori))
        return feats

    def forward(self, im, px, py, pxyz, pknn, num_points):
        feats = self.forward_2d_features(im)
        masks3d = self.kpclassifier(feats, px, py, pxyz, pknn, num_points)
        return masks3d
