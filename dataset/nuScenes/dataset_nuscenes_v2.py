# Copyright 2022 - Valeo Comfort and Driving Assistance - Spyros Gidaris @ valeo.ai
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

import json
import os
import numpy as np

from pathlib import Path
from torch.utils import data


map_name_from_segmentation_class_to_segmentation_index = {
    'ignore': 0,
    'barrier': 1,
    'bicycle': 2,
    'bus': 3,
    'car': 4,
    'construction_vehicle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'trailer': 9,
    'truck': 10,
    'driveable_surface': 11,
    'other_flat': 12,
    'sidewalk': 13,
    'terrain': 14,
    'manmade': 15,
    'vegetation': 16
}


class Nuscenes(data.Dataset):
    def __init__(self, dataroot, version='v1.0-trainval', split='train'):
        assert version in ['v1.0-trainval', 'v1.0-mini']
        assert split in ['train', 'val']
        self.version = version
        self.split = split
        self.dataroot = dataroot

        info_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'nuscenes_lidar_n_label_data_info.json')
        with open(info_file_path, 'r') as f:
            info_data = json.load(f)

        self.lidar_filenames, self.label_filenames = info_data[version][split]
        class_mapper = info_data["general_to_segmentation_index"]
        class_mapper = {int(key): val for (key, val) in class_mapper.items()}
        self.general_to_segmentation_index = np.vectorize(class_mapper.__getitem__)

        self.mapped_cls_name = {}
        for v, k in map_name_from_segmentation_class_to_segmentation_index.items():
            self.mapped_cls_name[k] = v

        print(f'nuscenes: {version} - {self.split} #samples: {len(self.lidar_filenames)}')

    def __len__(self):
        return len(self.lidar_filenames)

    def loadDataByIndex(self, index):
        lidar_path = os.path.join(self.dataroot, self.lidar_filenames[index])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.split == 'test':
            sem_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            lidarseg_path = os.path.join(self.dataroot, self.label_filenames[index])
            sem_label = np.fromfile(lidarseg_path, dtype=np.uint8).reshape((-1, 1))

        pointcloud = raw_data[:, :4]
        inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label

    def labelMapping(self, sem_label):
        sem_label = self.general_to_segmentation_index(sem_label)
        assert sem_label.shape[-1] == 1
        sem_label = sem_label[:, 0]
        return sem_label
