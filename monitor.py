# coding:utf-8
# @Time         : 2019/5/23 
# @Author       : xuyouze
# @File Name    : monitor.py

import pynvml
import time
from train import train
from test import test_method

pynvml.nvmlInit()  # 这里的0是GPU
deviceCount = pynvml.nvmlDeviceGetCount()
batch_size = 200
while True:
    available_count_list = []
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print("device:{}:{},current memory:{}m/{}m".format(i, str(pynvml.nvmlDeviceGetName(handle)),
                                                           int(meminfo.used / 1000000), int(meminfo.total / 1000000)))
        # if meminfo.used / meminfo.total < 0.3:
        if meminfo.used / meminfo.total < 0.1 and len(available_count_list) < 2:
            available_count_list.append(i)
    if len(available_count_list) > 0:
        break
    time.sleep(30)
print("current used gpu {}".format(available_count_list))
print("train begin : {}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
begin_time = time.time()
train(str(available_count_list).replace("[", "").replace("]", ""), batch_size)
# test_method(str(available_count_list).replace("[", "").replace("]", ""))
print("end begin : {},total cost {} h".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                                              (time.time() - begin_time) / 3600))

pynvml.nvmlShutdown()
