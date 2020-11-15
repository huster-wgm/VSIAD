import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ResYNetSync(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode):
        super(ResYNetSync, self).__init__()
        kernels = [64, 128, 256, 512]
        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.conv1 = nn.Conv2d(3, kernels[0], kernel_size=7, 
                               stride=2, padding=3,
                               bias=False)
        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.bn1 = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.bn1 = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.bn1 = nn.GroupNorm(32, kernels[0])
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # Left arm
        self.deconvL10 = deconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL11 = deconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL12 = deconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL13 = deconv(kernels[0],  kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # Right arm
        self.deconvR10 = deconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResR1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvR11 = deconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResR2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvR12 = deconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResR3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvR13 = deconv(kernels[0], kernels[0], ratio="x2")
        self.predR = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='IN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size//2)**2/float(2*sigma**2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()
        
    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

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
        
        ## stage 1
        # left
        x7L = self.deconvL10(x6)
        # right
        x7R = self.deconvR10(x6)
        # sycn
        x7L, x7R = self._status_sync(x7L, x7R)
        
        ## stage 2
        # left
        x8L = x7L + x5 # add x7 and x5
        x9L = self.upResL1(x8L)
        x10L = self.deconvL11(x9L)
        # right
        x8R = x7R + x5 # add x7 and x5
        x9R = self.upResR1(x8R)
        x10R = self.deconvR11(x9R)
        # sycn
        x10L, x10R = self._status_sync(x10L, x10R)
        
        ## stage 3
        # left
        x11L = x10L + x4 # add x11 and x4
        x12L = self.upResL2(x11L)
        x13L = self.deconvL12(x12L)
        # right
        x11R = x10R + x4 # add x11 and x4
        x12R = self.upResR2(x11R)
        x13R = self.deconvR12(x12R)
        # sycn
        x13L, x13R = self._status_sync(x13L, x13R)

        ## stage 3
        # left
        x14L = x13L + x3 # add x13 and x3
        x15L = self.upResL3(x14L)
        x16L = self.deconvL13(x15L)
        # right
        x14R = x13R + x3 # add x13 and x3
        x15R = self.upResR3(x14R)
        x16R = self.deconvR13(x15R)
        # sycn
        x16L, x16R = self._status_sync(x16L, x16R)
        
        ## stage 4
        x_L = self.predL(x16L)
        x_R = self.predR(x16R)
        return x_L, x_R


def res18ynetsync(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        BNmode (str) in [ BN, IN, GN ]
    """
    model = ResYNetSync(BasicBlock, [2, 2, 2, 2],
                        src_ch, tar_ch, BNmode=BNmode)
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


def res34ynetsync(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        BNmode (str) in [ BN, IN, GN ]
    """
    model = ResYNetSync(BasicBlock, [3, 4, 6, 3],
                        src_ch, tar_ch, BNmode=BNmode)
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
    parser.add_argument('-tar1_ch', type=int, default=1,
                        help='nb channel of target 1')
    parser.add_argument('-tar2_ch', type=int, default=2,
                        help='nb channel of target 2')
    parser.add_argument('-base_kernel', type=int, default=12,
                        help='batch_size for training ')
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='learning rate for discriminator')
    args = parser.parse_args()

    x = torch.FloatTensor(
        np.random.random((args.base_kernel, args.src_ch, args.img_row, args.img_col)))

    for BNmode in ['BN', 'IN', 'GN']:
        generator = res18ynetsync(args.src_ch, [args.tar1_ch, args.tar2_ch], True, BNmode)
        gen_L, gen_R = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("res18ynetsync : BN=>{}".format(BNmode))
        print(" Network L-output : ", gen_L.shape)
        print(" Network R-output : ", gen_R.shape)
        print(" Params: {:0.1f}M".format(total_params / (10**6)))

        generator = res34ynetsync(args.src_ch, [args.tar1_ch, args.tar2_ch], True, BNmode)
        gen_L, gen_R = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("res34ynetsync : BN=>{}".format(BNmode))
        print(" Network L-output : ", gen_L.shape)
        print(" Network R-output : ", gen_R.shape)
        print(" Params: {:0.1f}M".format(total_params / (10**6)))

