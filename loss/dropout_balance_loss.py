# coding:utf-8
# @Time         : 2019/9/18 
# @Author       : xuyouze
# @File Name    : dropout_balance_loss.py


import torch
import torch.nn as nn

from config import TrainConfig
import numpy as np
from loss.registry import Loss

__all__ = ["DropoutBalanceLoss"]


@Loss.register("dropout_balance")
class DropoutBalanceLoss(nn.Module):
    # def __init__(
    #         self, bins=10,
    #         momentum=0,
    #         use_sigmoid=True,
    #         loss_weight=1.0):
    def __init__(self, config: TrainConfig):
        super(DropoutBalanceLoss, self).__init__()

        self.bins = config.bins
        self.edges = [float(x) / self.bins for x in range(self.bins + 1)]
        self.edges[-1] += 1e-6
        # self.outlier_sample_weight = None
        # self.difficult_sample_weight = torch.FloatTensor(40).fill_(1)
        self.config = config
        self.dropout_rate = torch.ones(1)
        self.dropout_scope = config.dropout_scope
        # self.easy_attr_lowest_lr = config.easy_attr_lowest_lr
        # self.hard_attr_lowest_lr = config.hard_attr_lowest_lr
        # self.loss_lr = torch.tensor(config.loss_lr)

        # self.hard_attr_dropout_lr = config.hard_dropout_lr
        # self.easy_attr_dropout_lr = config.easy_dropout_lr
        # self.easy_hard_attr_gap = config.easy_hard_attr_gap

        if self.config.continue_train:
            # self.easy_attr_dropout_lr = self.easy_attr_lowest_lr
            # self.hard_attr_dropout_lr = self.hard_attr_lowest_lr
            self.dropout_rate = torch.zeros(1)
            self.dropout_scope = self.config.dropout_scope_lowest
        self.weight_num = torch.zeros(40, self.bins)

    def forward(self, pred, target, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            multi class target(index) for each sample.
        """
        edges = self.edges
        weights = torch.ones_like(pred)
        batch_size, class_size = target.shape
        g = torch.abs(pred.sigmoid() - target).detach()

        # generate rand_matrix
        rand_mat = torch.FloatTensor(batch_size, class_size).uniform_(0, 1).cuda()

        loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target).sum(0)
        ln_loss = torch.log10(1 + loss)
        norm_loss = 5 - 10 * (ln_loss - ln_loss.min()) / (ln_loss.max() - ln_loss.min())
        sigmoid_norm_rate = torch.sigmoid(norm_loss)
        dropout_rate = torch.where(sigmoid_norm_rate > self.dropout_rate.cuda(), sigmoid_norm_rate,
                                   self.dropout_rate.cuda())
        # dropout method
        idxs = (g >= edges[self.bins - self.dropout_scope]) & (g < edges[self.bins])
        drop_idxs = rand_mat.gt(dropout_rate).float()
        weights = weights - drop_idxs * idxs.float()

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
        # loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight.cuda())(pred, target)
        # loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target)
        return loss.mean()

    def update_loss_dropout(self, epoch):
        # easy_lr = max(1.0 - max(0, epoch - self.config.niter) / float(self.config.dropout_decay + 1),
        #               self.easy_attr_lowest_lr)
        # hard_lr = max(1.0 - max(0, epoch - self.config.niter) / float(self.config.dropout_decay + 1),
        #               self.hard_attr_lowest_lr)

        # lr_l = 1.0 - max(0, epoch - self.config.niter) / float(self.config.niter_decay + 1)

        # self.loss_lr = self.config.loss_lr * lr_l

        # self.easy_attr_dropout_lr = self.config.easy_dropout_lr * easy_lr
        # self.hard_attr_dropout_lr = self.config.hard_dropout_lr * hard_lr
        # if self.config.continue_train:
        #     self.easy_attr_dropout_lr = self.easy_attr_lowest_lr
        #     self.hard_attr_dropout_lr = self.hard_attr_lowest_lr

        # print("loss lr= {}, easy_attr_dropout_lr = {}, hard_attr_dropout_lr= {}".format(self.loss_lr,
        #                                                                                 self.easy_attr_dropout_lr,
        #                                                                                 self.hard_attr_dropout_lr))
        #
        # dropout_range
        self.dropout_scope = int(max(self.config.dropout_scope - epoch * (
                self.config.dropout_scope - self.config.dropout_scope_lowest) / self.config.dropout_scope_decay,
                                     self.config.dropout_scope_lowest))
        #
        epoch = self.config.dropout_decay if epoch > self.config.dropout_decay else epoch
        self.dropout_rate = max(torch.cos(torch.tensor(np.pi * epoch / (self.config.dropout_decay * 2))),
                                torch.zeros(1))

        # ellipse
        # self.dropout_rate = torch.tensor(np.sqrt(1 - np.power(epoch / self.config.dropout_decay, 2)))
        # self.dropout_rate = max(1.0 - epoch / self.config.dropout_decay, 0)

        print("dropout_rate= {},dropout_range={}".format(self.dropout_rate, self.dropout_scope))

    def get_weight_num(self):
        a = self.weight_num
        self.weight_num = torch.zeros(40, self.bins)
        return a
