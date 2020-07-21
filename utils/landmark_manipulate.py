# coding:utf-8
# @Time         : 2019/5/13 
# @Author       : xuyouze
# @File Name    : landmark_manipulate.py

from config.celebA_train_config import *
from PIL import Image, ImageDraw
import numpy as np
import os

# 读取landmark文件
# 这两张图片的label 出现错误
# 184195.jpg 84  105   94  104   91  109   88  116   95  114
# 062183.jpg [63, 128, 124, 105, 73, 137, 71, 144, 85, 139]
config = TrainConfig()

image_dir = os.path.join(config.data_root_dir, config.img_dir)
landmark_file_name = os.path.join(config.data_root_dir, config.landmark_file)
with open(landmark_file_name, "r") as f:
    landmark = np.zeros((202600, 10))
    image_names = []
    f.readline()
    f.readline()
    lines = f.readlines()
    index = 1
    for line in lines:
        vals = line.split()
        image_names.append(vals[0])
        for j in range(10):
            # change the labels
            # self.attr[id, j] = int(vals[j + 1])
            landmark[index, j] = vals[j + 1]
        index += 1

for i in [62183, 184195]:
    image = Image.open(os.path.join(image_dir, str(i).zfill(6) + ".jpg")).convert('RGB')
    left_eye = np.array([landmark[i][0], landmark[i][1]])
    right_eye = np.array([landmark[i][2], landmark[i][3]])
    nose = np.array([landmark[i][4], landmark[i][5]])
    left_mouth = np.array([landmark[i][6], landmark[i][7]])
    right_mouth = np.array([landmark[i][8], landmark[i][9]])
    eye = (left_eye + right_eye) / 2
    mouth = (left_mouth + right_mouth) / 2
    d = np.linalg.norm(mouth - eye)

    # 这个参数需要注意
    h = w = 2.5 * d
    # 根据landmark 切割 bounding box
    # 这个 比例需要再做几次实验进行验证
    region = image.crop((landmark[i][0] - 0.45 * w if landmark[i][0] - 0.45 * w > 0 else 0,
                         landmark[i][1] - 0.6 * h if landmark[i][1] - 0.6 * h > 0 else 0,
                         landmark[i][-2] + 0.45 * w if landmark[i][-2] + 0.45 * w < image.size[0] else image.size[0],
                         landmark[i][-1] + 0.5 * h if landmark[i][-1] + 0.5 * h < image.size[1] else image.size[1]))
    # 画点
    draw = ImageDraw.Draw(image)
    draw.point([(landmark[i][0], landmark[i][1])], fill=(255, 0, 0))
    draw.point([(landmark[i][2], landmark[i][3])], fill=(255, 0, 0))
    draw.point([(landmark[i][4], landmark[i][5])], fill=(255, 0, 0))
    draw.point([(landmark[i][6], landmark[i][7])], fill=(255, 0, 0))
    draw.point([(landmark[i][8], landmark[i][9])], fill=(255, 0, 0))
    # im_upper.show()
    # im_middle.show()
    # im_lower.show()
    # region.show()

# calculate the max and mean size of segmentation
# upper = np.zeros([202600, 2])
# middle = np.zeros([202600, 2])
# lower = np.zeros([202600, 2])
# whole = np.zeros([202600, 2])
# print(im_rotate.size)
# image_size[i] = im_lower.size
# upper[i] = im_upper.size
# middle[i] = im_middle.size
# lower[i] = im_lower.size
# whole[i] = im_rotate.size
# print("upper max {0}, mean {1}".format(np.max(upper[1:], axis=0), np.mean(upper[1:], axis=0)))
# print("middle max {0}, mean {1}".format(np.max(middle[1:], axis=0), np.mean(middle[1:], axis=0)))
# print("lower max {0}, mean {1}".format(np.max(lower[1:], axis=0), np.mean(lower[1:], axis=0)))
# print("whole max {0}, mean {1}".format(np.max(whole[1:], axis=0), np.mean(whole[1:], axis=0)))

# f1 = plt.figure(1)
# plt.subplot(221)
#
# plt.scatter(upper[1:, 0], upper[1:, 1])
# plt.subplot(222)
# plt.scatter(middle[1:, 0], middle[1:, 1])
#
# plt.subplot(223)
# plt.scatter(lower[1:, 0], lower[1:, 1])
#
# plt.subplot(224)
# plt.scatter(whole[1:, 0], whole[1:, 1])
#
# plt.show()
