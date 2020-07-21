# coding:utf-8
# @Time         : 2019/12/9 
# @Author       : xuyouze
# @File Name    : moon_loss.py


import torch
import torch.nn as nn

from config import TrainConfig
import numpy as np

from loss.registry import Loss

__all__ = ["Moon"]


@Loss.register("moon")
class Moon(nn.Module):
    def __init__(self, config: TrainConfig):
        super(Moon, self).__init__()

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
        pos_gt_idx = (self.pos_num > self.neg_num).astype(float)
        weights = torch.ones_like(pred)
        batch_size, class_size = target.shape
        target = target.cpu()

        for i in range(class_size):
            # print(target[:, i].numpy())
            minority_idx = np.array(
                [j[0] for j in np.argwhere(target[:, i].numpy() != pos_gt_idx[i])])
            if len(minority_idx) > 0:
                if pos_gt_idx[i]:
                    weights[minority_idx, i] *= self.pos_num[i] / self.neg_num[i]
                else:
                    weights[minority_idx, i] *= self.neg_num[i] / self.pos_num[i]
            # print(weights[:, i])
        target = target.cuda()

        loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target) * weights
        return loss.mean()
