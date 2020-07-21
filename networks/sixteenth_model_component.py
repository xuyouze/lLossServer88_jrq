# coding:utf-8
# @Time         : 2019/6/26 
# @Author       : xuyouze
# @File Name    : sixteenth_model_component.py


from torch import nn

from .model_component import init_net

__all__ = ["define_net"]


def define_net(init_type="normal", init_gain=0.02):
    # net_upper = FaceUpperNet()
    # net_middle = FaceMiddleNet()
    # net_lower = FaceLowerNet()
    net_whole = FaceWholeNet()
    return init_net(net_whole, init_type, init_gain)


class FaceWholeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input size is 178 * 218 * 3
        # output size of net is
        # attribute number is 12
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.linear = nn.Linear(1024, 12)
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 40)
        self.model = [
            ConvBlock(3, 32),
            self.max_pooling,

            ConvBlock(32, 32),
            ConvBlock(32, 64),
            self.max_pooling,

            ConvBlock(64, 64),
            ConvBlock(64, 128),
            self.max_pooling,

            ConvBlock(128, 128),
            ConvBlock(128, 256),
            self.max_pooling,

            ConvBlock(256, 256),
            ConvBlock(256, 512),
            self.max_pooling,

            ConvBlock(512, 512),
            ConvBlock(512, 1024),
            self.max_pooling,
            nn.AdaptiveAvgPool2d(1),
        ]

        self.net = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return self.linear2(x)


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
