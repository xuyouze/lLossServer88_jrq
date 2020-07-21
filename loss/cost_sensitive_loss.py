# coding:utf-8
# @Time         : 2019/12/9 
# @Author       : xuyouze
# @File Name    : cost_sensitive_loss.py


import torch
import torch.nn as nn

from config import TrainConfig
import numpy as np

from loss.registry import Loss

__all__ = ["CostSensitive"]


@Loss.register("cost_sensitive")
class CostSensitive(nn.Module):
    def __init__(self, config: TrainConfig):
        super(CostSensitive, self).__init__()

        self.config = config

    def forward(self, pred, target, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            multi class target(index) for each sample.
        """
        self.pos_num = self.config.dataset_config.global_attr_pos_num
        self.neg_num = self.config.dataset_config.global_attr_neg_num
        total_num = self.pos_num[0] + self.neg_num[0]
        pos_gt_idx = (self.pos_num > self.neg_num).astype(float)

        weights = torch.ones_like(pred).cpu()
        batch_size, class_size = target.shape
        target = target.cpu()

        for i in range(class_size):
            majority_idx = np.array(
                [j[0] for j in np.argwhere(target[:, i].numpy() == pos_gt_idx[i])])
            minority_idx = np.array(
                [j[0] for j in np.argwhere(target[:, i].numpy() != pos_gt_idx[i])])
            if pos_gt_idx[i]:
                weights[majority_idx, i] *= self.pos_num[i]
                weights[minority_idx, i] *= self.neg_num[i]
            else:
                weights[majority_idx, i] *= self.neg_num[i]
                weights[minority_idx, i] *= self.pos_num[i]
            weights[:, i] = np.exp(-1 / total_num * weights[:, i])
        weights = weights.cuda()
        target = target.cuda()

        loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target) * weights
        return loss.mean()
