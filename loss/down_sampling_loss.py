# coding:utf-8
# @Time         : 2019/12/9 
# @Author       : xuyouze
# @File Name    : down_sampling.py


import torch
import torch.nn as nn

from config import TrainConfig
import numpy as np

from loss.registry import Loss

__all__ = ["DownSampling"]


@Loss.register("down_sampling")
class DownSampling(nn.Module):
    def __init__(self, config: TrainConfig):
        super(DownSampling, self).__init__()

        self.config = config

    def forward(self, pred, target, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            multi class target(index) for each sample.
        """
        loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target).cpu()
        weights = torch.zeros_like(pred)
        batch_size, class_size = target.shape
        target = target.cpu()
        pos_sum = target.sum(0)
        neg_sum = batch_size - pos_sum
        pos_gt_idx = (pos_sum >= neg_sum).float().numpy()

        for i in range(class_size):
            majority_idx = np.array(
                [j[0] for j in np.argwhere(target[:, i].numpy() == pos_gt_idx[i])])
            minority_idx = np.array(
                [j[0] for j in np.argwhere(target[:, i].numpy() != pos_gt_idx[i])])
            weights[majority_idx[loss[majority_idx, i].argsort(descending=True)][0:len(minority_idx)], i] = 1
            if len(minority_idx) > 0:
                weights[minority_idx, i] = 1

        target = target.cuda()

        loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target) * weights
        return loss.mean()
