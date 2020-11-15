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


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet(nn.Module):

    def __init__(self, block, layers, src_ch=1, tar_ch=3, instance=False):
        super(ResNet, self).__init__()
        self.src_ch = src_ch
        if isinstance(tar_ch, list):
            tar_ch = sum(tar_ch)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64) if not instance else nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, instance=instance)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, instance=instance)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, instance=instance)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, instance=instance)

        # Left arm
        self.deconvL10 = deconv(512, 256, ratio="x2")
        self.inplanes = 256
        self.upResL1 = self._make_layer(block, 256, layers[2], instance=instance)
        self.deconvL11 = deconv(256, 128, ratio="x2")
        self.inplanes = 128
        self.upResL2 = self._make_layer(block, 128, layers[1], instance=instance)
        self.deconvL12 = deconv(128,  64, ratio="x2")
        self.inplanes = 64
        self.upResL3 = self._make_layer(block, 64, layers[0], instance=instance)
        self.deconvL13 = deconv( 64,  64, ratio="x2")
        self.predL = nn.Conv2d(64, tar_ch, kernel_size=3, stride=1, padding=1, bias=False)

        # # Right arm
        # self.deconvR10 = deconv(512, 256, ratio="x2")
        # self.inplanes = 256
        # self.upResR1 = self._make_layer(block, 256, layers[2], instance=instance)
        # self.deconvR11 = deconv(256, 128, ratio="x2")
        # self.inplanes = 128
        # self.upResR2 = self._make_layer(block, 128, layers[1], instance=instance)
        # self.deconvR12 = deconv(128,  64, ratio="x2")
        # self.inplanes = 64
        # self.upResR3 = self._make_layer(block, 64, layers[0], instance=instance)
        # self.deconvR13 = deconv( 64,  64, ratio="x2")
        # self.predR = nn.Conv2d(64, tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, instance=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion) if not instance else nn.InstanceNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward network
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x2 = self.relu(x1)
        # x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        # generate left
        x7L = self.deconvL10(x6)
        # x8L = x7L + x5 # add x7 and x5
        x9L = self.upResL1(x7L)
        x10L = self.deconvL11(x9L)
        # x11L = x10L + x4 # add x11 and x4
        x12L = self.upResL2(x10L)
        x13L = self.deconvL12(x12L)
        # x14L = x13L + x3 # add x13 and x3
        x15L = self.upResL3(x13L)
        x16L = self.deconvL13(x15L)
        x_L = self.predL(x16L)
        
        # # generate right
        # x7R = self.deconvR10(x6)
        # x8R = x7R + x5 # add x7 and x5
        # x9R = self.upResR1(x8R)
        # x10R = self.deconvR11(x9R)
        # x11R = x10R + x4 # add x11 and x4
        # x12R = self.upResR2(x11R)
        # x13R = self.deconvR12(x12R)
        # x14R = x13R + x3 # add x13 and x3
        # x15R = self.upResR3(x14R)
        # x16R = self.deconvR13(x15R)
        # x_R = self.predR(x16R)
        return x_L


def res18net(src_ch, tar_ch, pretrained=False, instance=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],
                    src_ch, tar_ch, instance=instance)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def res34net(src_ch, tar_ch, pretrained=False, instance=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3],
                    src_ch, tar_ch, instance=instance)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet34'])
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

    for inst in [True, False]:
        generator = res18net(args.src_ch, args.tar_ch, True, inst)
        gen_y = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("res18net{}".format("inst =>" if inst else " =>"))
        print(" Network output : ", gen_y.shape)
        print(" Params: {:0.1f}M".format(total_params / (10**6)))

        generator = res34net(args.src_ch, args.tar_ch, True, inst)
        gen_y = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("res34net{}".format("inst =>" if inst else " =>"))
        print(" Network output : ", gen_y.shape)
        print(" Params: {:0.1f}M".format(total_params / (10**6)))
