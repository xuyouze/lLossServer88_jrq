# coding:utf-8
# @Time         : 2019/9/26 
# @Author       : xuyouze
# @File Name    : clothing_config.py

from .registry import DatasetConfig


@DatasetConfig.register("clothing")
class ClothingConfig(object):

    def __init__(self):
        super(ClothingConfig, self).__init__()

        # dataset parameters
        self.data_root_dir = "/media/data1/xuyouze/Deep-Fashion"
        self.part_file = 'Eval/list_eval_partition.txt'
        self.attr_file = 'Anno/list_attr_img.txt'

        self.img_dir = "Img"
        self.img_size = 256
        self.crop_size = 224
        self.attribute_num = 1000
