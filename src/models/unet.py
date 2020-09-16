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


class UNet(nn.Module):
    """
    with VGG16 backend
    """
    def __init__(self,
                 src_ch=3,
                 tar_ch=1,
                 base_kernel=64,):
        super(UNet, self).__init__()
        if isinstance(tar_ch, list):
            tar_ch = sum(tar_ch)
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        # down&pooling
        self.downblock1 = UNetDownx2(
            src_ch, kernels[0])
        self.maxpool1 = nn.MaxPool2d(2)

        self.downblock2 = UNetDownx2(
            kernels[0], kernels[1])
        self.maxpool2 = nn.MaxPool2d(2)

        self.downblock3 = UNetDownx2(
            kernels[1], kernels[2])
        self.maxpool3 = nn.MaxPool2d(2)

        self.downblock4 = UNetDownx3(
            kernels[2], kernels[3])
        self.maxpool4 = nn.MaxPool2d(2)

        # center convolution
        self.center = ConvBlock(kernels[3], kernels[4])

        # up&concating
        self.upblock4 = UNetUpx3(
            kernels[4], kernels[3])

        self.upblock3 = UNetUpx2(
            kernels[3], kernels[2])

        self.upblock2 = UNetUpx2(
            kernels[2], kernels[1])

        self.upblock1 = UNetUpx2(
            kernels[1], kernels[0])

        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], tar_ch, 1))

    def forward(self, x):
        dx11 = self.downblock1(x)
        dx12 = self.maxpool1(dx11)

        dx21 = self.downblock2(dx12)
        dx22 = self.maxpool2(dx21)

        dx31 = self.downblock3(dx22)
        dx32 = self.maxpool3(dx31)

        dx41 = self.downblock4(dx32)
        dx42 = self.maxpool4(dx41)

        cx = self.center(dx42)

        ux4 = self.upblock4(cx, dx41)
        ux3 = self.upblock3(ux4, dx31)
        ux2 = self.upblock2(ux3, dx21)
        ux1 = self.upblock1(ux2, dx11)

        return self.outconv1(ux1)


def unet(src_ch, tar_ch, pretrained=False, backend=False, **kwargs):
    """Constructs a fcn8s model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = UNet(src_ch, tar_ch)
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
        generator = unet(args.src_ch, args.tar_ch, pre)
        gen_y = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("UNet => ")
        print(" Network output : ", gen_y.shape)
        print(" Params: {:0.1f}M".format(total_params / (10**6)))