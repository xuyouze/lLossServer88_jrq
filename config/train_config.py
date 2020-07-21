# coding:utf-8
# @Time         : 2019/9/16 
# @Author       : xuyouze
# @File Name    : train_config.py


from .base_config import BaseConfig

__all__ = ["TrainConfig"]


class TrainConfig(BaseConfig):
    def __init__(self):
        super(TrainConfig, self).__init__()

        # network saving and print parameters
        self.save_latest_freq = 256 * 1024  # frequency of saving the latest results
        self.save_epoch_freq = 1  # frequency of saving checkpoints at the end of epochs
        self.save_by_iter = False  # whether saves model by iteration
        self.print_freq = 200  # frequency of showing training results on console
        self.continue_train = False  # continue training: load the latest model
        self.isTrain = True

        # training parameters
        self.epoch_start = 0  # the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...
        self.niter = 0  # of iter at starting learning rate used for linear learning rate policy
        self.niter_decay = 60  # of iter to linearly decay learning rate to zero,

        self.beta1 = 0.9  # momentum term of adam
        self.batch_size = 128

        # lr parameters
        self.lr = 0.001  # initial learning rate for adam
        # self.lr = 0.00002  # initial learning rate for adam
        # self.lr_policy = "linear"  # learning rate policy. [linear | step | plateau | cosine | warm_up]
        self.lr_policy = "warm_up"  # learning rate policy. [linear | step | plateau | cosine | warm_up]
        # self.lr_policy = "cosine"  # learning rate policy. [linear | step | plateau | cosine | warm_up]

        # loss parameters
        self.gamma = 2
        self.alpha = 1
        self.size_average = True

        self.loss_lr = 1

        self.bins = 1000

        self.easy_dropout_lr = 1
        self.hard_dropout_lr = 1

        self.easy_attr_lowest_lr = 0.3
        self.hard_attr_lowest_lr = 0
        self.easy_hard_attr_gap = 0.05
        self.dropout_decay = 10
        # self.dropout_decay = 0

        self.dropout_scope = 500
        self.dropout_scope_decay = 10
        self.dropout_scope_lowest = 300
