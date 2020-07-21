# coding:utf-8
# @Time         : 2019/10/9 
# @Author       : xuyouze
# @File Name    : dukemtmc_config.py
import numpy as np
from .registry import DatasetConfig


@DatasetConfig.register("duke")
class DukeMTMCConfig(object):

    def __init__(self):
        super(DukeMTMCConfig, self).__init__()

        # dataset parameters
        self.data_root_dir = "/media/data2/xuyouze/DukeMTMC-reID"
        self.data_group_suffix = ['bounding_box_train', 'query', 'bounding_box_test']
        self.attr_file = "attribute/duke_attribute.mat"
        self.attribute_num = 23
        self.global_attr_pos_num = np.array([
            9560,
            2887,
            682,
            4131,
            6883,
            3392,
            2444,
            2701,
            9879,
            1239,
            953,
            221,
            1810,
            1483,
            501,
            314,
            6887,
            1170,
            305,
            2036,
            5295,
            27,
            802])
        self.global_attr_neg_num = np.array([
            6962,
            13635,
            15840,
            12391,
            9639,
            13130,
            14078,
            13821,
            6643,
            15283,
            15569,
            16301,
            14712,
            15039,
            16021,
            16208,
            9635,
            15352,
            16217,
            14486,
            11227,
            16495,
            15720

        ])
