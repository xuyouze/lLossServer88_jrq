# coding:utf-8
# @Time         : 2019/5/15 
# @Author       : xuyouze
# @File Name    : base_config.py
import importlib
import os
import sys
import torch

import logging

from .dataset_config import build_dataset_config
from .logger_config import config

__all__ = ["BaseConfig"]


class BaseConfig(object):
    def __init__(self):
        # model component parameters
        self.checkpoints_dir = "/media/data2/xuyouze/ckp_server"

        self.dataset_name = "celebA"
        # self.dataset_name = "lfwa"
        # self.dataset_name = "clothing"
        # self.dataset_name = "duke"
        # self.dataset_name = "market"
        self.model_name = "common"
        # self.model_name = "loss-first"
        # self.model_name = "loss-second"
        # self.model_name = "loss-third"

        # self.network_name = "common"
        # self.network_name = "sixteenth"
        self.network_name = "resnet"
        # self.network_name = "xiaokang"
        # self.network_name = "eighteenth"
        # self.loss_name = "focal"
        # self.loss_name = "ghmc"
        # self.loss_name = "bce"
        # self.loss_name = "dropout"
        # self.loss_name = "balance_drop"
        self.loss_name = "dropout_balance"
        # self.loss_name = "balance"
        # self.loss_name = "down_sampling"
        # self.loss_name = "over_sampling"
        # self.loss_name = "moon"
        # self.loss_name = "comparison"
        # self.loss_name = "cost_sensitive"

        # self.loss_name = "cost_sensitive"
        # self.loss_name = "balance"
        # self.loss_name = "pos-neg-balance"

        self.init_type = "normal"  # network initialization [normal | xavier | kaiming | orthogonal]
        self.init_gain = 0.2  # scaling factor for normal, xavier and orthogonal.
        # global saving and loading parameters
        self.batch_size = 100
        self.num_threads = 4

        self.last_epoch = "last"
        self.load_iter = 0
        self.isTrain = None

        self.dataset_config = build_dataset_config(self.dataset_name)
        self.balance_attr_pos_prop = torch.FloatTensor([0.5] * self.dataset_config.attribute_num)

        # logging
        logging.config.dictConfig(config)

        self.logger = logging.getLogger("TrainLogger")
        self.test_logger = logging.getLogger("TestLogger")

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)