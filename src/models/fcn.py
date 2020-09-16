#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .block import *


__all__ = [
    'vgg16', 'vgg19',
]


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGGbackend(nn.Module):
    def __init__(self,
                 src_ch=3,
                 base_kernel=64):
        super(VGGbackend, self).__init__()
        self.src_ch = src_ch
        kernels = [base_kernel * i for i in [1, 2, 4, 8, 16]]

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.src_ch, kernels[0], 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[0], kernels[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[1], kernels[1], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(kernels[1], kernels[2], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[2], kernels[2], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[2], kernels[2], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(kernels[2], kernels[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[3], kernels[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[3], kernels[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),)

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(kernels[3], kernels[4], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[4], kernels[4], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[4], kernels[4], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        return conv5, conv4, conv3


class FCN32s(nn.Module):
    def __init__(self,
                 src_ch=3,
                 tar_ch=1,
                 base_kernel=64,):
        super(FCN32s, self).__init__()
        kernels = [base_kernel * i for i in [1, 2, 4, 8, 16]]

        self.backend = VGGbackend(src_ch, base_kernel)
        if isinstance(tar_ch, list):
            tar_ch = sum(tar_ch)
        self.classifier = nn.Sequential(
            nn.Conv2d(kernels[4], 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, tar_ch, 1),)

#         # generate output
#         self.outconv1 = nn.Sigmoid() if tar_ch == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        conv5, conv4, conv3 = self.backend(x)

        score = self.classifier(conv5)
        # up = F.upsample(score, x.size()[2:], mode='bilinear')
        up = F.interpolate(score, x.size()[2:], mode='bilinear')
        return up


class FCN16s(nn.Module):
    def __init__(self,
                 src_ch=3,
                 tar_ch=1,
                 base_kernel=64,):
        super(FCN16s, self).__init__()
        kernels = [base_kernel * i for i in [1, 2, 4, 8, 16]]

        self.backend = VGGbackend(src_ch, base_kernel)
        if isinstance(tar_ch, list):
            tar_ch = sum(tar_ch)
        self.classifier = nn.Sequential(
            nn.Conv2d(kernels[4], 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, tar_ch, 1),)

        self.score_pool4 = nn.Conv2d(kernels[3], tar_ch, 1)

#         # generate output
#         self.outconv1 = nn.Sigmoid() if tar_ch == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        conv5, conv4, conv3 = self.backend(x)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)

        # score = F.upsample(score, score_pool4.size()[2:], mode='bilinear')
        score = F.interpolate(score, score_pool4.size()[2:], mode='bilinear')
        score += score_pool4
        # up = F.upsample(score, x.size()[2:], mode='bilinear')
        up = F.interpolate(score, x.size()[2:], mode='bilinear')
        return up


class FCN8s(nn.Module):
    def __init__(self,
                 src_ch=3,
                 tar_ch=1,
                 base_kernel=64,):
        super(FCN8s, self).__init__()
        kernels = [base_kernel * i for i in [1, 2, 4, 8, 16]]

        self.backend = VGGbackend(src_ch, base_kernel)
        if isinstance(tar_ch, list):
            tar_ch = sum(tar_ch)
        self.classifier = nn.Sequential(
            nn.Conv2d(kernels[4], 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, tar_ch, 1),)

        self.score_pool4 = nn.Conv2d(kernels[3], tar_ch, 1)
        self.score_pool3 = nn.Conv2d(kernels[2], tar_ch, 1)

#         # generate output
#         self.outconv1 = nn.Sigmoid() if tar_ch == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        conv5, conv4, conv3 = self.backend(x)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)

        # score = F.upsample(score, score_pool4.size()[2:], mode='bilinear')
        score = F.interpolate(score, score_pool4.size()[2:], mode='bilinear') 
        score += score_pool4
        # score = F.upsample(score, score_pool3.size()[2:], mode='bilinear')
        score = F.interpolate(score, score_pool3.size()[2:], mode='bilinear')
        score += score_pool3
        # up = F.upsample(score, x.size()[2:], mode='bilinear')
        up = F.interpolate(score, x.size()[2:], mode='bilinear')
        return up


def fcn32s(src_ch, tar_ch, pretrained=False, backend=False, **kwargs):
    """Constructs a fcn8s model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FCN8s(src_ch, tar_ch)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['vgg16'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def fcn16s(src_ch, tar_ch, pretrained=False, backend=False, **kwargs):
    """Constructs a fcn8s model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FCN16s(src_ch, tar_ch)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['vgg16'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def fcn8s(src_ch, tar_ch, pretrained=False, backend=False, **kwargs):
    """Constructs a fcn8s model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FCN8s(src_ch, tar_ch)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['vgg16'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


if __name__ == "__main__":
    # Hyper Parameters
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-img_row', type=int, default=224,
                        help='img_row of input')
    parser.add_argument('-img_col', type=int, default=224,
                        help='img_col of input ')
    parser.add_argument('-src_ch', type=int, default=1,
                        help='nb channel of source')
    parser.add_argument('-tar_ch', type=int, default=3,
                        help='nb channel of target')
    args = parser.parse_args()

    x = torch.FloatTensor(
        np.random.random((1, args.src_ch, args.img_row, args.img_col)))

    for pre in [True, False]:
        generator = fcn32s(args.src_ch, args.tar_ch, pre, pre)
        gen_y = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("FCN32s => ")
        print(" Network output : ", gen_y.shape)
        print(" Params: {:0.1f}M".format(total_params / (10**6)))

        generator = fcn16s(args.src_ch, args.tar_ch, pre, pre)
        gen_y = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("FCN16s => ")
        print(" Network output : ", gen_y.shape)
        print(" Params: {:0.1f}M".format(total_params / (10**6)))

        generator = fcn8s(args.src_ch, args.tar_ch, pre, pre)
        gen_y = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("FCN8s => ")
        print(" Network output : ", gen_y.shape)
        print(" Params: {:0.1f}M".format(total_params / (10**6)))

