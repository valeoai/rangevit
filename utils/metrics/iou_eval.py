'''
The MIT License
Copyright (c) 2019 Tiago Cortinhal (Halmstad University, Sweden), George Tzelepis (Volvo Technology AB, Volvo Group Trucks Technology, Sweden) and Eren Erdal Aksoy (Halmstad University and Volvo Technology AB, Sweden)
Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.

References:
https://github.com/PRBonn/lidar-bonnetal
https://github.com/TiagoCortinhal/SalsaNext
'''

import numpy as np
import torch


class IOUEval:
    def __init__(self, n_classes, device=torch.device('cpu'), ignore=None, is_distributed=False):
        self.n_classes = n_classes
        self.device = device
        # if ignore is larger than n_classes, consider no ignoreIndex
        self.ignore = torch.tensor(ignore).long()
        self.include = torch.tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]).long()
        print('[IOU EVAL] IGNORE: ', self.ignore)
        print('[IOU EVAL] INCLUDE: ', self.include)
        self.is_distributed = is_distributed
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.n_classes, self.n_classes), device=self.device).long()
        self.ones = None
        self.last_scan_size = None  # for when variable scan size is used

    def addBatch(self, x, y):  # x=preds, y=targets
        # if numpy, pass to pytorch to tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x)).long().to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).long().to(self.device)

        # sizes should be 'batch_size x H x W'
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # idxs are labels and predictions
        idxs = torch.stack([x_row, y_row], dim=0)

        # ones is what I want to add to conf when I
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()
            self.last_scan_size = idxs.shape[-1]

        # make confusion matrix (cols = gt, rows = pred)
        self.conf_matrix = self.conf_matrix.index_put_(
            tuple(idxs), self.ones, accumulate=True)

    def getStats(self):
        # remove fp and fn from confusion on the ignore classes cols and rows
        conf = self.conf_matrix.clone().double()
        if self.is_distributed:
            conf_gpu = conf.cuda()
            torch.distributed.barrier()
            torch.distributed.all_reduce(conf_gpu)
            conf = conf_gpu.to(self.conf_matrix)
            torch.distributed.barrier()
            del conf_gpu
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns 'iou mean', 'iou per class' ALL CLASSES

    def getIoUnAcc(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()

        total = tp + fp + 1e-15
        acc = tp / total
        acc_mean = acc[self.include].mean()

        return iou_mean, iou, acc_mean, acc  # returns 'iou mean', 'iou per class' ALL CLASSES

    def getAcc(self):
        tp, fp, fn = self.getStats()
        total = tp + fp + 1e-15
        acc = tp / total
        acc_mean = acc[self.include].mean()
        return acc_mean, acc

    def getRecall(self):
        tp, fp, fn = self.getStats()
        total = tp + fn + 1e-15
        recall = tp / total
        recall_mean = recall[self.include].mean()
        return recall_mean, recall
