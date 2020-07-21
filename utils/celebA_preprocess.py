# coding:utf-8
# @Time    : 2019/5/9 
# @Author  : xuyouze
# @File Name    : celebA_preprocess.py


"""
对CelebA图片进行预处理
---------------------
使用AFFACT内的公式对图片进行裁剪, 剪出脸部并输出大小
对AFFACT 的参数进行了修改
将脸部切为三个部分,并输出对应的尺寸
对每个碎片进行padding
"""
from math import ceil, floor
from PIL import Image, ImageOps
import numpy as np
import os
import time
from config.celebA_base_config import CelebABaseConfig

start_time = time.time()
config = CelebABaseConfig()

# read the align landmark file
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
            landmark[index, j] = vals[j + 1]
        index += 1
print("landmark file read completed,cost {}s".format(time.time() - start_time))
# read the image
image_dir = os.path.join(config.data_root_dir, config.img_dir)

for i in range(1, len(image_names) + 1):
    # for i in range(1, 100):
    image = Image.open(os.path.join(image_dir, image_names[i - 1])).convert('RGB')
    left_eye = np.array([landmark[i][0], landmark[i][1]])
    right_eye = np.array([landmark[i][2], landmark[i][3]])
    nose = np.array([landmark[i][4], landmark[i][5]])
    left_mouth = np.array([landmark[i][6], landmark[i][7]])
    right_mouth = np.array([landmark[i][8], landmark[i][9]])

    # compute the distance between eye and mouth

    eye = (left_eye + right_eye) / 2
    mouth = (left_mouth + right_mouth) / 2
    d = np.linalg.norm(mouth - eye)

    # 这个参数需要注意
    h = w = 2.5 * d
    # if right_eye[0] - left_eye[0] != 0:
    #     alpha_eye = np.arctan((right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0]))
    # else:
    #     alpha_eye = 0
    # if right_mouth[0] - right_mouth[0] != 0:
    #     alpha_mouth = np.arctan((right_mouth[1] - left_mouth[1]) / (right_mouth[0] - right_mouth[0]))
    # else:
    #     alpha_mouth = 0

    # 直接使用原图进行剪切
    region = image

    # 计算切割的y
    eye_y = left_eye[1] if left_eye[1] < right_eye[1] else right_eye[1]
    mouth_y = left_mouth[1] if left_mouth[1] > right_mouth[1] else right_mouth[1]

    # 对图片进行切割
    im_upper = region.crop([0, 0, region.size[0], nose[1]])
    im_middle = region.crop([0, eye_y, region.size[0], mouth_y])
    im_lower = region.crop([0, nose[1], region.size[0], region.size[1]])

    # 对图片进行padding
    im_upper = ImageOps.expand(im_upper, border=(0, 0, 0, config.face_whole_y - im_upper.size[1]), fill=0)

    im_middle = ImageOps.expand(im_middle, border=(0,
                                                   floor((config.face_whole_y - im_middle.size[1]) / 2), 0,
                                                   ceil((config.face_whole_y - im_middle.size[1]) / 2)
                                                   ), fill=0)
    im_lower = ImageOps.expand(im_lower, border=(0, config.face_whole_y - im_lower.size[1], 0, 0), fill=0)

    # save whole, upper, middle, lower face image
    region.save(os.path.join(config.data_root_dir, os.path.join(config.face_whole, image_names[i - 1])))
    im_upper.save(os.path.join(config.data_root_dir, os.path.join(config.face_upper, image_names[i - 1])))
    im_middle.save(os.path.join(config.data_root_dir, os.path.join(config.face_middle, image_names[i - 1])))
    im_lower.save(os.path.join(config.data_root_dir, os.path.join(config.face_lower, image_names[i - 1])))
    if i % 1000 == 0:
        print("[{}/{}],total cost {}s".format(i, 202600, time.time() - start_time))
