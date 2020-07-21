# coding:utf-8
# @Time         : 2019/10/8 
# @Author       : xuyouze
# @File Name    : balance_loss.py


import torch
import torch.nn as nn

from config import TrainConfig
import numpy as np

from loss.registry import Loss

__all__ = ["BalanceLoss"]


@Loss.register("balance")
class BalanceLoss(nn.Module):
    # def __init__(
    #         self, bins=10,
    #         momentum=0,
    #         use_sigmoid=True,
    #         loss_weight=1.0):
    def __init__(self, config: TrainConfig):
        super(BalanceLoss, self).__init__()

        self.config = config
        self.bins = config.bins
        self.edges = [float(x) / self.bins for x in range(self.bins + 1)]
        self.edges[-1] += 1e-6

    def forward(self, pred, target, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            multi class target(index) for each sample.
        """
        weights = torch.ones_like(pred)
        batch_size, class_size = target.shape
        edges = self.edges
        g = torch.abs(pred.sigmoid() - target).detach()

        # selective learning method, according to the 0.5*batch
        batch_current_size = weights.sum(0).cpu()
        balance_num = self.config.balance_attr_pos_prop * batch_current_size

        pos_sum = (target * weights).sum(0).cpu()
        neg_sum = batch_current_size - pos_sum
        pos_gt_idx = (pos_sum >= balance_num)
        neg_gt_idx = (neg_sum > balance_num)

        target = target.cpu()
        g = g.cpu()
        # for i in hard_attr_idx:
        easy_sample_idxs = ((g >= edges[0]) & (g < edges[10]))
        for i in range(class_size):
            majority_idx = target[:, i] == pos_gt_idx[i].float()
            majority_easy = easy_sample_idxs[:, i] & majority_idx
            weights[majority_easy, i] = 0
            majority_idx = np.array([j[0] for j in np.argwhere(target[:, i].numpy() == pos_gt_idx[i].float().numpy())])

            weights[majority_idx, i] *= balance_num[i] / len(majority_idx)

            minority_idx = np.array([j[0] for j in np.argwhere(target[:, i].numpy() == neg_gt_idx[i].float().numpy())])
            if len(minority_idx) > 0:
                weights[minority_idx, i] *= (batch_current_size[i] - balance_num[i]) / len(minority_idx)
        target = target.cuda()

        loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target) * weights
        return loss.mean()
