# coding:utf-8
# @Time         : 2019/7/29 
# @Author       : xuyouze
# @File Name    : resnet_model_component.py
import torch
from torch import nn
from torchvision.models import resnet50

from networks.registry import Network

__all__ = ["define_net"]


@Network.register("resnet")
def define_net(config):
    net_whole = Resnet50(True, config.dataset_config.attribute_num)
    return torch.nn.DataParallel(net_whole).cuda()


class Resnet50(nn.Module):
    """
    change a resnet into 40 sub-branches architecture
    """

    def __init__(self, pre_train, output_num=40):
        super().__init__()
        pre_model = resnet50(pretrained=pre_train)
        # self.resnet_layer = nn.Sequential(*list(pre_model.children())[:-1])
        # self.Linear_layer = nn.Linear(2048, output_num, bias=False)
        # self.BN_layer = nn.BatchNorm2d(2048)
        """--------------------------------------------------------------------"""
        self.shared_res_layer = nn.Sequential(
            *list(pre_model.children())[:-1],
            nn.BatchNorm2d(2048)
        )
        self.private_linear_layer1 = [nn.Linear(2048, 512) for i in range(output_num)]
        self.private_linear_layer2 = [nn.Linear(512, 2) for i in range(output_num)]

    def forward(self, x):
        # x = self.resnet_layer(x)
        # # nn.Conv2d(kernel_size=3, stride=1, padding=0, bias=False)
        # x = self.BN_layer(x)
        #
        # x = x.view(x.size(0), -1)
        # x = self.Linear_layer(x)
        # return x
        x = self.shared_res_layer(x)  # extract common features
        x_bl1 = [self.private_linear_layer1[i](x) for i in range(len(self.private_linear_layer1))]  # expand 40 branches
        x_bl2 = [self.private_linear_layer2[i](x_bl1[i]) for i in range(len(self.private_linear_layer2))]
        return x_bl2
