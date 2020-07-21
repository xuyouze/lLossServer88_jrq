# coding:utf-8
# @Time         : 2019/10/9 
# @Author       : xuyouze
# @File Name    : dukemtmc_config.py
import numpy as np
from .registry import DatasetConfig


@DatasetConfig.register("market")
class MarketConfig(object):

    def __init__(self):
        super(MarketConfig, self).__init__()

        # dataset parameters
        self.data_root_dir = "/media/data1/xuyouze/Market-1501"
        self.data_group_suffix = ['bounding_box_train', 'query', 'bounding_box_test']
        self.attr_file = "attribute/market_attribute.mat"
        self.attribute_num = 30

        self.global_attr_pos_num = np.array([213,
                                             10101,
                                             2436,
                                             186,
                                             3576,
                                             3122,
                                             1546,
                                             11055,
                                             8007,
                                             12293,
                                             4164,
                                             293,
                                             5378,
                                             1711,
                                             3913,
                                             1569,
                                             498,
                                             641,
                                             1440,
                                             785,
                                             1139,
                                             4950,
                                             1091,
                                             548,
                                             29,
                                             208,
                                             2067,
                                             1874,
                                             219,
                                             1390])
        self.global_attr_neg_num = np.array([
            12723,
            2835,
            10500,
            12750,
            9360,
            9814,
            11390,
            1881,
            4929,
            643,
            8772,
            12643,
            7558,
            11225,
            9023,
            11367,
            12438,
            12295,
            11496,
            12151,
            11797,
            7986,
            11845,
            12388,
            12907,
            12728,
            10869,
            11062,
            12717,
            11546

        ])
