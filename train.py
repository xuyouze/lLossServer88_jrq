# coding:utf-8
# @Time         : 2019/5/14 
# @Author       : xuyouze
# @File Name    : train.py
import os
import time

import numpy as np
import torch

from config import TrainConfig, TestConfig
from models import *
from dataset import *



def validate(model):
    print("--------------------------------------------------------")
    print("test the model using the validate dataset")

    validate_config = TestConfig()
    validate_config.isTest = True
    validate_dataset = create_dataset(validate_config)
    model.eval()
    model.clear_precision()
    model.set_validate_size(len(validate_dataset))
    print("validate dataset len: %d " % len(validate_dataset))
    validate_total_iter = int(len(validate_dataset) / validate_config.batch_size)
    for j, valida_data in enumerate(validate_dataset):
        model.set_input(valida_data)
        print("[%s/%s]" % (j, validate_total_iter))
        model.test()
    print(model.get_model_precision())
    print(model.get_model_class_balance_precision())
    # print(model.get_model_precision() / 100)
    # [print("%0.4f" % i.item()) for i in model.get_model_precision()]
    print("mean accuracy: {}".format(torch.mean(model.get_model_precision())))
    print("class_balance accuracy: {}".format(torch.mean(model.get_model_class_balance_precision())))
    # print("mean accuracy: {}".format(torch.mean(model.get_model_precision() / 100)))
    print("validate mode end")
    print("--------------------------------------------------------")


def train(gpu, batch_size):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # set_seed(2019)

    config = TrainConfig()
    config.continue_train = False  # continue training: load the latest model
    # config.continue_train = True  # continue training: load the latest model
    # config.load_iter = 12

    config.batch_size = batch_size

    print("current batch size is {}".format(config.batch_size))
    dataset = create_dataset(config)  # create a dataset given opt.dataset_mode and other options
    model = create_model(config)
    model.setup()
    config.print_freq = int(len(dataset) / config.batch_size / 10)
    total_iters = 0
    print("dataset len: %d " % len(dataset))

    for epoch in range(config.epoch_start, config.niter + config.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        print("epoch [{}/{}] begin at: {} ,learning rate : {}".format(epoch, config.niter + config.niter_decay,
                                                                      time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                    time.localtime(epoch_start_time)),
                                                                      model.get_learning_rate()))
        for i, data in enumerate(dataset):
            iter_start_time = time.time()

            total_iters += config.batch_size
            epoch_iter += config.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if i % config.print_freq == 0:
                losses = model.get_current_loss()
                t_comp = (time.time() - iter_start_time) / config.batch_size
                print("epoch[%d/%d], iter[%d/%d],current loss=%s,Time consuming: %s sec" % (
                    epoch, config.niter + config.niter_decay, epoch_iter, len(dataset), losses, t_comp))

            if total_iters % config.save_latest_freq == 0:
                print("saving the last model (epoch %d, total iters %d)" % (epoch, total_iters))
                model.save_networks(config.last_epoch)

        # if epoch % config.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks(config.last_epoch)
        model.save_networks("iter_%d" % epoch)
        validate(model)
        # get the model precision
        # result = model.get_criterion().numpy()
        # np.savetxt("result.txt", result, delimiter=',')
        # model.set_loss_dropout(epoch)

        model.train()
        model.update_loss_dropout(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, config.niter + config.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.
