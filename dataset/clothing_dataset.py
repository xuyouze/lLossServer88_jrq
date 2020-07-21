# coding:utf-8
# @Time         : 2019/9/26 
# @Author       : xuyouze
# @File Name    : clothing_dataset.py

import os

from PIL import Image
from torchvision.transforms import transforms

from config import *
from dataset.registry import Dataset
from .base_dataset import BaseDataset
import numpy as np

__all__ = ["ClothingDataset"]


def get_img_attr(attr_file):
    # attr = np.zeros((289222, 1000))
    attr = {}
    with open(attr_file) as f:
        f.readline()
        f.readline()
        lines = f.readlines()

        for i, line in enumerate(lines):
            vals = line.split()
            attr[vals[0]] = [0 if vals[j + 1] == "-1" else 1 for j in range(1000)]
    return attr


def get_img_name_by_partition(part_dir, partition_flag):
    img = []
    with open(part_dir) as f:
        f.readline()
        f.readline()
        lines = f.readlines()
        for line in lines:
            pic_dir, status = line.split()
            if status == partition_flag:
                img.append(pic_dir)
    return img


@Dataset.register("clothing")
class ClothingDataset(BaseDataset):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        self.config = config
        self.dataset_config = config.dataset_config
        self.attr_file = os.path.join(self.dataset_config.data_root_dir, self.dataset_config.attr_file)
        self.attr = get_img_attr(self.attr_file)
        self.partition_file = os.path.join(self.dataset_config.data_root_dir, self.dataset_config.part_file)

        if config.isTrain:
            self.image_names = get_img_name_by_partition(self.partition_file, "train")
            self.transform = transforms.Compose([
                transforms.Resize(self.dataset_config.img_size),
                transforms.RandomResizedCrop(self.dataset_config.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.dataset_config.crop_size),
                transforms.CenterCrop(self.dataset_config.crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            if self.config.isTest:
                self.image_names = get_img_name_by_partition(self.partition_file, "test")
            else:
                self.image_names = get_img_name_by_partition(self.partition_file, "val")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_dir = os.path.join(self.config.dataset_config.data_root_dir, self.config.dataset_config.img_dir)

        img = Image.open(os.path.join(img_dir, self.image_names[index])).convert('RGB')

        img_transform = self.transform(img)
        return img_transform, np.array(self.attr[self.image_names[index]])
