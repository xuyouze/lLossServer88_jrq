# coding:utf-8
# @Time         : 2019/6/10 
# @Author       : xuyouze
# @File Name    : image_size_manipulate.py

import torch
from torch import nn


class ConvBlock(nn.Module):

    def __init__(self, input_channel, output_channel, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.net = [
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(output_channel),
            nn.PReLU()
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


x = torch.randn((16, 3, 178, 218))
max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# m = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=0)
model = [
    nn.Conv2d(3,3,3,stride=1, padding=0, dilation=2)
    # ConvBlock(3, 32),
    # max_pooling,
    #
    # ConvBlock(32, 32),
    # ConvBlock(32, 64),
    # max_pooling,
    #
    # ConvBlock(64, 64),
    # ConvBlock(64, 128),
    # max_pooling,
    #
    # ConvBlock(128, 128),
    # ConvBlock(128, 256),
    # max_pooling,
    #
    # ConvBlock(256, 256),
    # ConvBlock(256, 512),
    # max_pooling,
    #
    # ConvBlock(512, 512),
    # ConvBlock(512, 1024),
    # max_pooling,
    # nn.AdaptiveAvgPool2d(1)
]

m = nn.Sequential(*model)
y = m(x)
print(y.shape)