import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random

seed = 1#seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)#numpy产生的随机数一致
random.seed(seed)

# CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
# 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
torch.backends.cudnn.deterministic = True

# 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
torch.backends.cudnn.benchmark = False


class ResNet50BasicBlock(nn.Module):
    def __init__(self, in_channel, outs, kernerl_size, stride, padding):
        super(ResNet50BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernerl_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernerl_size[1], stride=stride[0], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernerl_size[2], stride=stride[0], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(outs[2])

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)

        return F.relu(out + x)


class ResNet50DownBlock(nn.Module):
    def __init__(self, in_channel, outs, kernel_size, stride, padding):
        super(ResNet50DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(outs[2])

        self.extra = nn.Sequential(
            nn.Conv2d(in_channel, outs[2], kernel_size=1, stride=stride[3], padding=0),
            nn.BatchNorm2d(outs[2])
        )

    def forward(self, x):
        x_shortcut = self.extra(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        return F.relu(x_shortcut + out)


class ResNet50(nn.Module):
    def __init__(self, start_channels):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(start_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResNet50DownBlock(64, outs=[64, 64, 256], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        )

        self.layer2 = nn.Sequential(
            ResNet50DownBlock(256, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
            ResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            ResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            ResNet50DownBlock(512, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
        )

        self.layer3 = nn.Sequential(
            ResNet50DownBlock(512, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2],
                              padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            ResNet50DownBlock(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0]),
            ResNet50DownBlock(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0]),
            ResNet50DownBlock(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0])
        )

        self.layer4 = nn.Sequential(
            ResNet50DownBlock(1024, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2],
                              padding=[0, 1, 0]),
            ResNet50DownBlock(2048, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0]),
            ResNet50DownBlock(2048, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0])
        )

        # self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, ceil_mode=False)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.latent = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)

        # out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)
        return self.latent(out.squeeze())


class fusion_ori_grad(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.oriEmbed = ResNet50(start_channels=3)
        self.gradEmbed = ResNet50(start_channels=8)

        self.gender_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.MLP = nn.Sequential(
            nn.Linear(2048+32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )


    def forward(self, ori, grad, gender):
        ori_feature = self.oriEmbed(ori)
        grad_feature = self.gradEmbed(grad)
        gender_encode = self.gender_encoder(gender)

        return self.MLP(torch.cat((ori_feature, grad_feature, gender_encode), dim=-1))


class disGrad(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.oriEmbed = ResNet50(start_channels=3)

        self.gender_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )


        self.MLP = nn.Sequential(
            nn.Linear(1024 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, ori, gender):
        ori_feature = self.oriEmbed(ori)
        gender_encode = self.gender_encoder(gender)

        return self.MLP(torch.cat((ori_feature, gender_encode), dim=-1))



if __name__ == '__main__':
    ori = torch.randn(2, 1, 512, 512)
    grad = torch.randn(2, 8, 512, 512)
    gender = torch.randn(2, 1)
    net = fusion_ori_grad()
    # out = net(ori, grad, gender)
    # print('out.shape: ', out.shape)
    # print(out)

    print(sum(p.nelement() for p in net.parameters() if p.requires_grad==True)/1e6)