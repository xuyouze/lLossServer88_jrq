# coding:utf-8
# @Time         : 2019/8/27 
# @Author       : xuyouze
# @File Name    : umd_dataset.py

import os

from PIL import Image
from torchvision.transforms import transforms

from config import *
from dataset.registry import Dataset
from .base_dataset import BaseDataset
import numpy as np

__all__ = ["UmdDataset"]


def get_img_attr(attr_file):
    return np.loadtxt(attr_file)


def get_img_name(img_file):
    img = []
    with open(img_file, mode="r") as f:
        lines = f.readlines()
        img = [line.replace("\n", "") for line in lines]
    return img


@Dataset.register("umd")
class UmdDataset(BaseDataset):

    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self.config = config
        self.attr_file = os.path.join(config.data_root_dir, config.attr_file)
        self.img_name_file = os.path.join(config.data_root_dir, config.img_name_file)
        self.attr = get_img_attr(self.attr_file)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.image_names = get_img_name(self.img_name_file)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_dir = os.path.join(self.config.data_root_dir, self.config.img_dir)

        img = Image.open(os.path.join(img_dir, self.image_names[index])).convert('RGB')

        img = self.transform(img)

        img_id = int(self.image_names[index].split(".")[0]) - 1

        # return face_upper, face_middle, face_lower, face_whole, self.attr[id, :]
        return img, self.attr[img_id, :]
