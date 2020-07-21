# coding:utf-8
# @Time         : 2019/5/9
# @Author       : xuyouze
# @File Name    : celebA_config.py
import numpy as np

from .registry import DatasetConfig

@DatasetConfig.register("celebA")
class CelebAConfig(object):

    def __init__(self):
        super(CelebAConfig, self).__init__()

        self.data_root_dir = "/media/data1/xuyouze/CelebA_full"
        self.part_file = 'Eval/list_eval_partition.txt'
        # self.attr_file = 'Anno/list_attr_celeba.txt'
        # self.attr_file = 'Anno/check_attr.txt'
        self.attr_file = 'Anno/second_check_attr.txt'
        self.selfattr_file = 'Anno/second_check_attr.txt'
        # self.attr_file = 'Anno/outlier_correction_file.txt'
        self.landmark_file = "Anno/list_landmarks_align_celeba.txt"

        self.img_dir = "img_align_celeba"
        self.face_whole = "img_align_celeba"
        # self.face_upper = "face_upper"
        # self.face_middle = "face_middle"
        # self.face_lower = "face_lower"
        # global saving parameters

        # self.face_whole_y = 218
        # self.face_whole_x = 178
        self.attribute_num = 40
        self.global_attr_pos_num = np.array([22516., 54090., 103833., 41446., 4547., 30709., 48785.,
                                             47516., 48472., 29983., 10312., 41572., 28803., 11663.,
                                             9459., 13193., 12716., 8499., 78390., 92189., 84434.,
                                             97942., 8417., 23329., 169158., 57567., 8701., 56210.,
                                             16163., 13315., 11449., 97669., 42222., 64744., 38276.,
                                             9818., 95715., 24913., 14732., 156734.])

        self.global_attr_neg_num = np.array([144593,
                                             119492,
                                             79167,
                                             129490,
                                             159057,
                                             138085,
                                             123557,
                                             124429,
                                             123864,
                                             138503,
                                             154408,
                                             129578,
                                             139384,
                                             153381,
                                             155199,
                                             152249,
                                             152433,
                                             155874,
                                             100215,
                                             89125,
                                             94509,
                                             84284,
                                             156128,
                                             143901,
                                             26991,
                                             116669,
                                             155765,
                                             117924,
                                             149730,
                                             152245,
                                             153614,
                                             84690,
                                             128823,
                                             110788,
                                             132408,
                                             154731,
                                             86333,
                                             143006,
                                             150880,
                                             35982]
                                            )
