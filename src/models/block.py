import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3x3(in_planes, out_planes, stride=1):
    """ 3x3x3 convolution with padding """
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=(1, stride, stride),
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                     stride=stride, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution with padding """
    return nn.Conv3d(in_planes, out_planes, kernel_size=1,
                     stride=(1, stride, stride), bias=False)

def get_deconv_params(ratio="x2"):
    """
    args:
        ratio:(str) upsample level
        H_out =(H_in−1)×stride[0]−2×pad[0]+ksize[0]+output_padding[0] 
    """
    if ratio=="x2":
        kernel_size = 2
        stride=2
    elif ratio=="x4":
        kernel_size = 2
        stride=4
    elif ratio=="x8":
        kernel_size = 4
        stride=8
    output_padding = (stride-kernel_size)
    return kernel_size, stride, output_padding


def depthwiseDeconv(in_planes, out_planes, ratio="x2", depthwise=True):
    """depthwise deconv"""
    kernel_size, stride, output_padding = get_deconv_params(ratio)
    return nn.ConvTranspose2d(in_planes, out_planes, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=0, 
                              bias=False,
                              output_padding=output_padding,
                              groups=out_planes if depthwise else 1)


def deconv(in_planes, out_planes, ratio="x2"):
    """2d deconv"""
    kernel_size, stride, opad = get_deconv_params(ratio)
    # print("DECONV", kernel_size, stride, opad)
    return nn.ConvTranspose2d(in_planes, out_planes, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=0, 
                              bias=False,
                              output_padding=opad,
                              groups=1)


def deconv3d(in_planes, out_planes, ratio="x2"):
    """3d deconv"""
    kernel_size, stride, opad = get_deconv_params(ratio)
    # print("DECONV", kernel_size, stride, opad)
    return nn.ConvTranspose3d(in_planes, out_planes, 
                              kernel_size=(1, kernel_size, kernel_size), 
                              stride=(1, stride, stride), 
                              padding=0, 
                              bias=False,
                              output_padding=opad,
                              groups=1)


def conv3x3bn(in_ch, out_ch, stride=1):
    "3x3 convolution with padding"
    convbn = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3,
                  stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),)
    return convbn


class Interp(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest', align_corners=None, utype='2D'):
        super(Interp, self).__init__()
        self.up = F.interpolate
        self.mode = mode
        self.align_corners = align_corners
        if utype == '2D':
            self.scale_factors = [scale_factor, scale_factor]
        elif utype == '3D':
            self.scale_factors = [1] + [scale_factor, scale_factor]
        else:
            raise ValueError('{} is not support'.format(utype))

    def forward(self, x):
        x = self.up(x, scale_factor=tuple(self.scale_factors), mode=self.mode, align_corners=self.align_corners)
        return x

    
def upconv(in_planes, out_planes, ratio="x2"):
    """2d upsampling"""
    return nn.Sequential(
        Interp(scale_factor=2, mode='nearest', align_corners=None, utype='2D'),
        conv1x1(in_planes, out_planes)
    )


def upconv3d(in_planes, out_planes, ratio="x2"):
    """3d upsampling"""
    return nn.Sequential(
        Interp(scale_factor=2, mode='nearest', align_corners=None, utype='3D'),
        conv1x1x1(in_planes, out_planes)
    )


class BasicBlockNB(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockNB, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(ConvBlock, self).__init__()
        # convolution block
        if is_bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),)
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)

    def forward(self, x):
        x = self.block(x)
        return x


class UNetDownx2(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetDownx2, self).__init__()
        # convolution block
        if is_bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),)
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)

    def forward(self, x):
        x = self.block(x)
        return x


class UNetDownx3(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetDownx3, self).__init__()
        # convolution block
        if is_bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),)
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)

    def forward(self, x):
        x = self.block(x)
        return x


class UNetUpx2(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv=False, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetUpx2, self).__init__()
        # upsampling and convolution block
        if is_deconv:
            self.upscale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        else:
            if is_bn:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.BatchNorm2d(out_ch),
                    Interp(scale_factor=2),)
            else:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    Interp(scale_factor=2),)
        self.block = UNetDownx2(in_ch, out_ch, is_bn, is_leaky, alpha)

    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.block(x)
        return x


class UNetUpx3(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv=False, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetUpx3, self).__init__()
        # upsampling and convolution block
        if is_deconv:
            self.upscale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        else:
            if is_bn:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.BatchNorm2d(out_ch),
                    Interp(scale_factor=2),)
            else:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    Interp(scale_factor=2),)
        self.block = UNetDownx3(in_ch, out_ch, is_bn, is_leaky, alpha)

    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.block(x)
        return x


class SegNetUpx2(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False):
        super(SegNetUpx2, self).__init__()
        # upsampling and convolution block
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.block = UNetDownx2(in_ch, out_ch, is_bn)

    def forward(self, x, indices, output_shape):
        x = self.unpool(x, indices, output_shape)
        x = self.block(x)
        return x


class SegNetUpx3(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False):
        super(SegNetUpx3, self).__init__()
        # upsampling and convolution block
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.block = UNetDownx3(in_ch, out_ch, is_bn)

    def forward(self, x, indices, output_shape):
        x = self.unpool(x, indices, output_shape)
        x = self.block(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, BNmode='GN'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        if BNmode == 'BN':
            self.bn1 = nn.BatchNorm2d(planes)
        elif BNmode == 'IN':
            self.bn1 = nn.InstanceNorm2d(planes)
        elif BNmode == 'GN':
            self.bn1 = nn.GroupNorm(32, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if BNmode == 'BN':
            self.bn2 = nn.BatchNorm2d(planes)
        elif BNmode == 'IN':
            self.bn2 = nn.InstanceNorm2d(planes)
        elif BNmode == 'GN':
            self.bn2 = nn.GroupNorm(32, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockNB3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockNB3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, instance=False):
        super(BasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes) if not instance else nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes) if not instance else nn.InstanceNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, instance=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes) if not instance else nn.InstanceNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes) if not instance else nn.InstanceNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if not instance else nn.InstanceNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, instance=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes) if not instance else nn.InstanceNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes) if not instance else nn.InstanceNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion) if not instance else nn.InstanceNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out