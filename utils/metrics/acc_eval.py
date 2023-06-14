'''
The MIT License
Copyright (c) 2019 Tiago Cortinhal (Halmstad University, Sweden), George Tzelepis (Volvo Technology AB, Volvo Group Trucks Technology, Sweden) and Eren Erdal Aksoy (Halmstad University and Volvo Technology AB, Sweden)
Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.

References:
https://github.com/PRBonn/lidar-bonnetal
https://github.com/TiagoCortinhal/SalsaNext
'''

import torch 

class AccEval(object):
    def __init__(self, topk=(1, ), is_distributed=False):
        self.topk = topk 
        self.is_distributed = is_distributed
    
    def getAcc(self, output, target):
        maxk = max(self.topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        if self.is_distributed:
            correct = correct.cuda()
            batch_size = torch.Tensor([batch_size]).cuda()
            torch.distributed.barrier()
            torch.distributed.all_reduce(correct)
            torch.distributed.all_reduce(batch_size)
            correct = correct.to(target)
            batch_size = batch_size.item()
        for k in self.topk:
            correct_k = correct[:k].float().sum()
            acc = correct_k.mul_(100.0/batch_size)
            res.append(acc)
        return res