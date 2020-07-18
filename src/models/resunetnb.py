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



class ResUNetNB(nn.Module):

    def __init__(self, block, layers, src_ch=1, tar_ch=3):
        super(ResUNetNB, self).__init__()
        self.src_ch = src_ch
        if isinstance(tar_ch, list):
            tar_ch = sum(tar_ch)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Left arm
        self.upconvL10 = upconv(512, 256, ratio="x2")
        self.inplanes = 256
        self.upResL1 = self._make_layer(block, 256, layers[2])
        self.upconvL11 = upconv(256, 128, ratio="x2")
        self.inplanes = 128
        self.upResL2 = self._make_layer(block, 128, layers[1])
        self.upconvL12 = upconv(128,  64, ratio="x2")
        self.inplanes = 64
        self.upResL3 = self._make_layer(block, 64, layers[0])
        self.upconvL13 = upconv( 64,  64, ratio="x2")
        self.predL = nn.Conv2d(64, tar_ch, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)
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
        x2 = self.relu(x1)
        # x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        # generate left
        x7L = self.upconvL10(x6)
        x8L = x7L + x5 # add x7 and x5
        x9L = self.upResL1(x8L)
        x10L = self.upconvL11(x9L)
        x11L = x10L + x4 # add x11 and x4
        x12L = self.upResL2(x11L)
        x13L = self.upconvL12(x12L)
        x14L = x13L + x3 # add x13 and x3
        x15L = self.upResL3(x14L)
        x16L = self.upconvL13(x15L)
        x_L = self.predL(x16L)
        
        return x_L


def res18unetNB(src_ch, tar_ch, pretrained=False, instance=True, **kwargs):
    """Constructs a ResUNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResUNetNB(BasicBlockNB, [2, 2, 2, 2],
                      src_ch, tar_ch)
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


def res34unetNB(src_ch, tar_ch, pretrained=False, instance=True, **kwargs):
    """Constructs a ResUNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResUNetNB(BasicBlockNB, [3, 4, 6, 3],
                      src_ch, tar_ch)
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
    parser.add_argument('-base_kernel', type=int, default=12,
                        help='batch_size for training ')
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='learning rate for discriminator')
    args = parser.parse_args()

    x = torch.FloatTensor(
        np.random.random((args.base_kernel, args.src_ch, args.img_row, args.img_col)))

    for inst in [True, False]:
        generator = res18unetNB(args.src_ch, args.tar_ch, True, inst)
        gen_y = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("res18unet{}".format("inst =>" if inst else " =>"))
        print(" Network output : ", gen_y.shape)
        print(" Params: {:0.1f}M".format(total_params / (10**6)))

        generator = res34unetNB(args.src_ch, args.tar_ch, True, inst)
        gen_y = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("res34unet{}".format("inst =>" if inst else " =>"))
        print(" Network output : ", gen_y.shape)
        print(" Params: {:0.1f}M".format(total_params / (10**6)))
