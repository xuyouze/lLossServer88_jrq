# coding:utf-8
# @Time         : 2019/8/27 
# @Author       : xuyouze
# @File Name    : UMD-AED-manipulate.py


import numpy as np
import os

data_root_dir = "/media/data1/xuyouze/UMD-UED"
attr_file = 'Anno/attr_anno_file'
img_dir = "croppedImages"
# -1 represents unmarked
# file = np.loadtxt(os.path.join(data_root_dir,attr_file))
attr_name = None
attr_file_dir = os.path.join(data_root_dir, attr_file)
img_file_dir = os.path.join(data_root_dir, img_dir)
attr = np.ones((2809, 40)) * -1
#
# with open('attr_name.txt', mode="r") as f:
#     attr_name = [line.replace("\n", "") for line in f]
#     os.mkdir("epoch_{}".format(j))
# for i in range(len(attr_name)):
#     print(attr_name[i].replace("_","").lower())
# file_name = "{}.txt".format(attr_name[i].replace("_", "").lower())
# attr_file_each = os.path.join(attr_file_dir, file_name)
# with open(attr_file_each) as f:
#     lines = f.readlines()
#     id = 0
#     for line in lines:
#         img_name, flag = line.split()
#         print("{}, {}".format(img_name, int(img_name.split(".")[0])))
# attr[int(img_name.split(".")[0]), i] = flag
# attr_anno_file = np.savetxt("attr_anno_file", attr[1:],fmt='%d')
#
# write img_file
# with open("img_file.txt", mode="w") as f:
#     for i in range(1, 2809):
#         f.writelines("{}.jpg\n".format("{}".format(i).zfill(4)))
with open('img_file.txt', mode="r") as f:
    lines = f.readlines()
    img = [line.replace("\n","") for line in lines]
    # print(img)