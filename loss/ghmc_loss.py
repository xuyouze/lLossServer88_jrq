# coding:utf-8
# @Time         : 2019/7/30
# @Author       : xuyouze
# @File Name    : ghmc_Loss.py
import numpy as np
import torch
import torch.nn as nn

from config import TrainConfig
from loss.registry import Loss

__all__ = ["GHMCLoss"]


@Loss.register("ghmc")
class GHMCLoss(nn.Module):
    # def __init__(
    #         self, bins=10,
    #         momentum=0,
    #         use_sigmoid=True,
    #         loss_weight=1.0):
    def __init__(self, config: TrainConfig):
        super(GHMCLoss, self).__init__()

        self.bins = 30
        self.momentum = 0.6
        self.edges = [float(x) / self.bins for x in range(self.bins + 1)]
        self.edges[-1] += 1e-6
        if self.momentum > 0:
            self.acc_sum = torch.zeros((config.dataset_config.attribute_num, self.bins))
        self.use_sigmoid = True
        self.loss_weight = 1

    def forward(self, pred, target, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            multi class target(index) for each sample.
        """

        edges = self.edges
        mmt = self.momentum
        weights = torch.ones_like(pred)
        batch_size, class_num = target.shape
        target = target.float()

        # gradient length
        g = torch.abs(pred.sigmoid() - target).detach()
        # tot = max(valid.float().sum().item(), 1.0)

        num_in_bin = torch.zeros((class_num, self.bins))
        n = np.zeros(class_num)
        for i in range(self.bins):
            # idxs = N * C
            # weight = N * C
            # num_in_bins = C * bins
            idxs = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin[:, i] = idxs.sum(dim=0)
            for j in range(class_num):
                if num_in_bin[j, i] > 0:
                    if mmt > 0:
                        self.acc_sum[j, i] = mmt * self.acc_sum[j, i] + (1 - mmt) * num_in_bin[j, i]
                        weights[idxs[:, j], j] = batch_size / self.acc_sum[j, i]
                    else:
                        weights[idxs[:i], j] = batch_size / num_in_bin[j, i]
                    n[j] += 1
        for j, k in enumerate(n):
            if k > 0:
                weights[:, j] = weights[:, j] / k

        loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target) * weights
        return loss.mean()
