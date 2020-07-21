# coding:utf-8
# @Time         : 2019/10/27 
# @Author       : xuyouze
# @File Name    : umd-test_model.py


import torch
from torch.autograd import Variable

from config import BaseConfig
from models.registry import Model
from models.base_model import BaseModel

__all__ = ["UmdTestModel"]


@Model.register("umd")
class UmdTestModel(BaseModel):
    def __init__(self, config: BaseConfig) -> None:
        super().__init__(config)
        # define the net, such as  net_%s % net_names
        self.config = config
        self.net_names = ["whole"]

        # define the net
        self.net_whole = self.create_network_model()

        # define input and output
        self.img_whole = self.attr = None
        self.output = self.output_whole = None

        # Define the attribute index for each part
        self.attr_whole_index = [i for i in range(config.dataset_config.attribute_num)]
        # define optimizer and loss

    def set_input(self, x):
        # self.img_upper, self.img_middle, self.img_lower, self.img_whole, self.attr = x
        self.img_whole, self.attr = x
        for name in self.net_names:
            setattr(self, "img_%s" % name, Variable(getattr(self, "img_%s" % name)).cuda())
            setattr(self, "attr_%s" % name,
                    Variable(self.attr[:, getattr(self, "attr_%s_index" % name)].cuda()).type(torch.cuda.FloatTensor))

    def forward(self):
        for name in self.net_names:
            setattr(self, "output_%s" % name, getattr(self, "net_%s" % name)(getattr(self, "img_%s" % name)))

    def test(self):
        with torch.no_grad():
            self.forward()
            # self.output[:, self.attr_upper_index] = self.output_upper
            self.output = torch.Tensor(self.attr.size(0), self.config.dataset_config.attribute_num)
            for name in self.net_names:
                self.output[:, getattr(self, "attr_%s_index" % name)] = getattr(self, "output_%s" % name).cpu()
            com1 = (self.output > 0.5).float()
            com2 = self.attr.float()
            result = (com1.eq(com2)).sum(0).float()
            self.accuracy.add_(result)
