# Copyright 2022 - Valeo Comfort and Driving Assistance
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
import numpy as np
from PIL import Image


class SemanticKitti(object):
    def __init__(self, root,  # directory where data is
                 sequences,  # sequences for this data (e.g. [1,3,4,6])
                 config_path,  # directory of config file
                 has_label=True):
        self.root = root
        self.sequences = sequences
        self.sequences.sort()  # sort seq id
        self.has_label = has_label

        # check file exists
        if os.path.isfile(config_path):
            self.data_config = yaml.safe_load(open(config_path, 'r'))
        else:
            raise ValueError(f'Config file not found: {config_path}')

        if os.path.isdir(self.root):
            print(f'Dataset found: {self.root}')
        else:
            raise ValueError(f'Dataset not found: {self.root}')

        self.pointcloud_files = []
        self.label_files = []
        for seq in self.sequences:
            # format seq id
            seq = '{0:02d}'.format(int(seq))
            print(f'parsing seq {seq}...')

            # get file list from path
            pointcloud_path = os.path.join(self.root, seq, 'velodyne')
            pointcloud_files = [
                os.path.join(pointcloud_path, f)
                for f in os.listdir(pointcloud_path) if '.bin' in f]

            if self.has_label:
                label_path = os.path.join(self.root, seq, 'labels')
                label_files = [
                    os.path.join(label_path, f)
                    for f in os.listdir(label_path) if '.label' in f]

            if self.has_label:
                assert (len(pointcloud_files) == len(label_files))

            self.pointcloud_files.extend(pointcloud_files)
            if self.has_label:
                self.label_files.extend(label_files)

        # sort for correspondance
        self.pointcloud_files.sort()
        if self.has_label:
            self.label_files.sort()

        print(f'Using {len(self.pointcloud_files)} pointclouds from sequences {self.sequences}')

        # load config -------------------------------------
        # get learning class map
        # map unused classes to used classes
        learning_map = self.data_config['learning_map']
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut[k] = v
        # learning map inv
        learning_map = self.data_config['learning_map_inv']
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut_inv = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut_inv[k] = v

        # compute ignore class by content ratio
        cls_content = self.data_config['content']
        content = np.zeros(len(self.data_config['learning_map_inv']), dtype=np.float32)
        for cl, freq in cls_content.items():
            x_cl = self.class_map_lut[cl]
            content[x_cl] += freq
        self.cls_freq = content

        self.mapped_cls_name = self.data_config['mapped_class_name']

    @staticmethod
    def readPCD(path):
        pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return pcd

    @staticmethod
    def readLabel(path):
        label = np.fromfile(path, dtype=np.int32)
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        return sem_label, inst_label

    def parsePathInfoByIndex(self, index):
        path = self.pointcloud_files[index]
        # linux path
        if '\\' in path:
            # windows path
            path_split = path.split('\\')
        else:
            path_split = path.split('/')
        seq_id = path_split[-3]
        frame_id = path_split[-1].split('.')[0]
        return seq_id, frame_id

    def labelMapping(self, label):
        label = self.class_map_lut[label]
        return label

    def loadLabelByIndex(self, index):
        sem_label, inst_label = self.readLabel(self.label_files[index])
        return sem_label, inst_label

    def loadDataByIndex(self, index):
        pointcloud = self.readPCD(self.pointcloud_files[index])
        if self.has_label:
            sem_label, inst_label = self.readLabel(self.label_files[index])
        else:
            sem_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
            inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label

    def __len__(self):
        return len(self.pointcloud_files)
