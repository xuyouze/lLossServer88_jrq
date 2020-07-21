# coding:utf-8
# @Time         : 2019/9/3 
# @Author       : xuyouze
# @File Name    : pos-neg-balance_loss.py


import torch
import torch.nn as nn

from config import TrainConfig
import numpy as np


class PosNegBalanceLoss(nn.Module):
    # def __init__(
    #         self, bins=10,
    #         momentum=0,
    #         use_sigmoid=True,
    #         loss_weight=1.0):
    def __init__(self, config: TrainConfig):
        super(PosNegBalanceLoss, self).__init__()

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
        g = torch.abs(pred.sigmoid() - target).detach().cpu()

        # dropout loss

        # generate rand_matrix
        rand_mat = torch.FloatTensor(batch_size, class_size).uniform_(0, 1).cuda()

        loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target).sum(0)
        ln_loss = torch.log10(1 + loss)
        norm_loss = 5 - 10 * (ln_loss - ln_loss.min()) / (ln_loss.max() - ln_loss.min())

        sigmoid_norm_rate = torch.sigmoid(norm_loss)

        self.dropout_rate = self.dropout_rate.cuda()
        dropout_rate = torch.where(sigmoid_norm_rate > self.dropout_rate, sigmoid_norm_rate, self.dropout_rate)

        # selecting learning method, according to the 0.5*batch
        # pos_sum = target.sum(0)
        # neg_sum = batch_size - pos_sum
        # balance_num = 0.5 * batch_size
        # pos_gt_idx = (pos_sum > balance_num).cpu()
        # neg_gt_idx = (neg_sum > balance_num).cpu()

        # dropout_num = torch.zeros_like(pos_sum).cpu()
        # dropout_num[pos_gt_idx] = pos_sum[pos_gt_idx].cpu() - balance_num
        # dropout_num[neg_gt_idx] = neg_sum[neg_gt_idx].cpu() - balance_num
        # dropout_num = dropout_num.cpu().int()

        # according to the global info to selecting dropout
        pos_sum = target.sum(0).cpu()
        neg_sum = batch_size - pos_sum
        balance_num = torch.zeros_like(pos_sum).cpu()
        balance_pos_num = self.config.global_attr_pos_prop * batch_size
        balance_neg_num = batch_size - balance_pos_num

        pos_gt_idx = (pos_sum > balance_pos_num).cpu()
        neg_gt_idx = (neg_sum > balance_neg_num).cpu()

        balance_num[pos_gt_idx] = balance_pos_num[pos_gt_idx]
        balance_num[neg_gt_idx] = balance_neg_num[neg_gt_idx]

        dropout_num = torch.zeros_like(pos_sum).cpu()
        dropout_num[pos_gt_idx] = pos_sum[pos_gt_idx] - balance_pos_num[pos_gt_idx]
        dropout_num[neg_gt_idx] = neg_sum[neg_gt_idx] - balance_neg_num[neg_gt_idx]
        dropout_num = dropout_num.int()

        hard_attr_idx = torch.FloatTensor(1, class_size).uniform_(0, 1) > dropout_rate.cpu()
        easy_attr_idx = np.where(hard_attr_idx == 0)[1]

        hard_attr_idx = np.where(hard_attr_idx == 1)[1]

        # hard_sample_idxs = (g >= edges[0]) & (g < edges[self.dropout_scope])
        target = target.cpu()
        # for i in range(class_size):
        for i in hard_attr_idx:
            majority_idx = np.array([j[0] for j in np.argwhere(target[:, i].numpy() == pos_gt_idx[i].float().numpy())])
            # g[majority_idx:,i]
            # weights[
            #     np.random.choice(a=majority_idx, size=dropout_num[i], replace=False), i] = 0
            # assign weight to majority class
            weights[majority_idx, i] *= balance_num[i] / len(majority_idx)

            # dropout weight according to g ( simple example)

            # weights[majority_idx[g[majority_idx, i].argsort()][0:dropout_num[i]], i] = 0

            # dropout weight according to g (outlier example)
            # weights[majority_idx[g[majority_idx, i].argsort()][0:dropout_num[i]], i] = 0
            # minority_idx = [j[0] for j in np.argwhere(target[:, i].numpy() == neg_gt_idx[i].float().numpy()) if
            #                 hard_sample_idxs[j, i] == 1]
            # minority_idx = np.array([j[0] for j in np.argwhere(target[:, i].numpy() == neg_gt_idx[i].float().numpy())])

            # if len(minority_idx) > 0:
            #     weights[minority_idx, i] *= balance_num[i] / len(minority_idx)
        for i in easy_attr_idx:
            majority_idx = np.array([j[0] for j in np.argwhere(target[:, i].numpy() == pos_gt_idx[i].float().numpy())])

            # weights[majority_idx, i] *= balance_num[i] / len(majority_idx)
            weights[majority_idx[g[majority_idx, i].argsort()][0:dropout_num[i]], i] = 0

            minority_idx = np.array([j[0] for j in np.argwhere(target[:, i].numpy() == neg_gt_idx[i].float().numpy())])
            if len(minority_idx) > 0:
                weights[minority_idx, i] *= (batch_size - balance_num[i]) / len(minority_idx)
        # dropout loss
        idxs = (g >= edges[self.bins - self.dropout_scope]) & (g < edges[self.bins])
        drop_idxs = rand_mat.gt(dropout_rate).float()

        weights = weights * (1 - drop_idxs * idxs.cuda().float())

        target = target.cuda()

        loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target) * weights
        return loss.mean()

    def update_loss_dropout(self, epoch):
        # self.dropout_scope = int(max(self.config.dropout_scope - epoch * (
        #         self.config.dropout_scope - self.config.dropout_scope_lowest) / self.config.dropout_decay,
        #                              self.config.dropout_scope_lowest))

        epoch = self.config.dropout_decay if epoch > self.config.dropout_decay else epoch
        self.dropout_rate = max(torch.cos(torch.tensor(np.pi * epoch / (self.config.dropout_decay * 2))),
                                torch.zeros(1))

        print("dropout_rate= {},dropout_range={}".format(self.dropout_rate, self.dropout_scope))

    def get_weight_num(self):
        a = self.weight_num
        self.weight_num = torch.zeros(40, self.bins)
        return a
