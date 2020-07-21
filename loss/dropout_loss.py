# coding:utf-8
# @Time         : 2019/7/30 
# @Author       : xuyouze
# @File Name    : dropout_loss.py


import torch
import torch.nn as nn

from config import TrainConfig
import numpy as np

from loss.registry import Loss

__all__ = ["DropoutLoss"]


@Loss.register("dropout")
class DropoutLoss(nn.Module):
    # def __init__(
    #         self, bins=10,
    #         momentum=0,
    #         use_sigmoid=True,
    #         loss_weight=1.0):
    def __init__(self, config: TrainConfig):
        super(DropoutLoss, self).__init__()
        self.bins = config.bins
        self.edges = [float(x) / self.bins for x in range(self.bins + 1)]
        self.edges[-1] += 1e-6
        self.config = config
        self.dropout_rate = torch.ones(1)
        self.dropout_scope = config.dropout_scope

        if self.config.continue_train:
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

        loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target) * weights
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

        self.config.logger.info("dropout_rate= {},dropout_range={}".format(self.dropout_rate, self.dropout_scope))

    def get_weight_num(self):
        a = self.weight_num
        self.weight_num = torch.zeros(40, self.bins)
        return a
