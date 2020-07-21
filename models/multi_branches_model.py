# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 8:34
# @Author  : JRQ
# @FileName: multi_branches_model.py

import torch
from torch import nn, optim
from torch.autograd import Variable

from config import BaseConfig
from models.registry import Model
from models.base_model import BaseModel
from loss import create_loss

__all__ = ["MultiBranchModel"]

Model.register("multi")


class MultiBranchModel(BaseModel):
    def __init__(self, config: BaseConfig):
        super(MultiBranchModel, self).__init__(config=config)
        self.config = config
        self.net_names = ["whole"]
        self.net_whole = self.create_network_model()
        self.attr_whole_index = [i for i in range(config.dataset_config.attribute_num)]
        for name in self.net_names:
            setattr(self, "img_%s" % name, None)
            setattr(self, "output_%s" % name, None)
            setattr(self, "attr_%s" % name, None)

            # define optimizer and loss
            if config.isTrain:
                for name in self.net_names:
                    if self.config.loss_name == "bce":
                        setattr(self, "criterion_%s" % name, nn.BCEWithLogitsLoss())
                    else:
                        setattr(self, "criterion_%s" % name, create_loss(self.config))

                    setattr(self, "optimizer_%s" % name,
                            optim.Adam(getattr(self, "net_%s" % name).parameters(), lr=config.lr,
                                       betas=(config.beta1, 0.999)))
                    setattr(self, "loss_%s" % name, None)
                    self.optimizers.append(getattr(self, "optimizer_%s" % name))
            else:
                self.correct = torch.FloatTensor(self.config.dataset_config.attribute_num).fill_(0)

    def set_input(self, x):
        self.img_whole, self.attr = x
        for name in self.net_names:
            setattr(self, "img_%s" % name, Variable(getattr(self, "img_%s" % name)).cuda())
            setattr(self, "attr_%s" % name,
                    Variable(self.attr[:, getattr(self, "attr_%s_index" % name)].cuda()).type(torch.cuda.FloatTensor))

    def forward(self):
        """
        for multi-branches situation, output is a list
        """
        for name in self.net_names:
            # self.fc_feature, self.output_whole = getattr(self, "net_%s" % name)(getattr(self, "img_%s" % name))
            setattr(self, "output_%s" % name, getattr(self, "net_%s" % name)(getattr(self, "img_%s" % name)))

    def backward(self):
        for name in self.net_names:
            loss = 0
            for j in range(len(getattr(self, "output_{}".format(name)))):
                loss += getattr(self, "criterion_{}".format(name))(getattr(self, "output_{}".format(name))[j],
                                                                   getattr(self, "attr_{}".format(name))[j].cuda())
            setattr(self, "loss_{}".format(name), loss)

            getattr(self, "loss_{}".format(name), loss).backward()

    def test(self):
        with torch.no_grad():
            self.forward()
            for name in self.net_names:
                for i in range(self.config.dataset_config.attribute_num):
                    pred_attr_i = getattr(self, "output_{}".format(name))[i].argmax(dim=1)
                    real_attr_i = getattr(self, "attr_{}".format(name))[:, i]
                    self.correct[i] += ((pred_attr_i == real_attr_i) + 0.0).sum()
                    self.pos_num[i] += pred_attr_i.sum().float()
                    tpr = (pred_attr_i * real_attr_i).sum().float()
                    tnr = ((1 - pred_attr_i) * (1 - real_attr_i)).sum().float()
                    self.tpr[i] += tpr
                    self.tnr[i] += tnr
