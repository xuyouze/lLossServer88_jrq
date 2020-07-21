# coding:utf-8
# @Time         : 2019/9/16 
# @Author       : xuyouze
# @File Name    : lfwa_config.py
import numpy as np

from .registry import DatasetConfig

@DatasetConfig.register("lfwa")

class LFWAConfig(object):

    def __init__(self):
        super(LFWAConfig, self).__init__()

        # dataset parameters
        self.data_root_dir = "/media/data1/xuyouze/LFW"
        # self.attr_train_file = 'anno/train.txt'
        self.attr_train_file = 'anno/lfwa_outlier_correction_file.txt'
        self.attr_test_file = 'anno/test.txt'
        self.img_dir = "img"
        self.img_augment_dir = "lfw-deepfunneled"
        self.attribute_num =40
        self.global_attr_pos_num = np.array([2571,
                                             1630,
                                             2314,
                                             3733,
                                             686,
                                             1034,
                                             2264,
                                             4335,
                                             788,
                                             259,
                                             971,
                                             2239,
                                             3337,
                                             2280,
                                             2257,
                                             1157,
                                             1482,
                                             1040,
                                             707,
                                             2128,
                                             4853,
                                             2057,
                                             758,
                                             4059,
                                             4520,
                                             3238,
                                             3216,
                                             4303,
                                             3655,
                                             1207,
                                             1962,
                                             2518,
                                             2371,
                                             2698,
                                             840,
                                             878,
                                             943,
                                             1258,
                                             4040,
                                             1002
                                             ])
        self.global_attr_neg_num = np.array([3692,
                                             4633,
                                             3949,
                                             2530,
                                             5577,
                                             5229,
                                             3999,
                                             1928,
                                             5475,
                                             6004,
                                             5292,
                                             4024,
                                             2926,
                                             3983,
                                             4006,
                                             5106,
                                             4781,
                                             5223,
                                             5556,
                                             4135,
                                             1410,
                                             4206,
                                             5505,
                                             2204,
                                             1743,
                                             3025,
                                             3047,
                                             1960,
                                             2608,
                                             5056,
                                             4301,
                                             3745,
                                             3892,
                                             3565,
                                             5423,
                                             5385,
                                             5320,
                                             5005,
                                             2223,
                                             5261
                                             ])
