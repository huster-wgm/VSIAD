#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import random
import numpy as np
import collections

import torch
from torchvision import transforms
from torchvision.transforms import functional as F


ImageNet_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))


class BrightShift(object):
    """
    Args:
        brightness (float) – How much to jitter brightness. brightness_factor is chosen uniformly from \
        [max(0, 1 - brightness), 1 + brightness].
     """

    def __init__(self, brightness):
        assert 0 <= brightness <= 1.0, "brightness should be in [0, 1.0]."
        self.brightness = brightness

    def __call__(self, img):
        bright = np.random.uniform(
            low=max(0, 1 - self.brightness), high=(1 + self.brightness))
        return F.adjust_brightness(img, bright)


class ContrastShift(object):
    """
    Args:
        contrast (float) – How much to jitter contrast. contrast_factor is chosen uniformly from \
        [max(0, 1 - contrast), 1 + contrast].
    """

    def __init__(self, contrast):
        assert 0 <= contrast <= 1.0, "contrast should be in [0, 1.0]."
        self.contrast = contrast

    def __call__(self, img):
        contrast = np.random.uniform(
            low=max(0, 1 - self.contrast), high=(1 + self.contrast))
        return F.adjust_contrast(img, contrast)


class SaturationShift(object):
    """
    Args:
        saturation (float) – How much to jitter saturation. saturation_factor is chosen uniformly from \
        [max(0, 1 - saturation), 1 + saturation].
    """

    def __init__(self, saturation):
        assert 0 <= saturation <= 1.0, "saturation should be in [0, 1.0]."
        self.saturation = saturation

    def __call__(self, img):
        saturation = np.random.uniform(
            low=max(0, 1 - self.saturation), high=(1 + self.saturation))
        return F.adjust_saturation(img, saturation)


class GammaShift(object):
    """
    Args:
        gamma (float) – Non negative real number, same as γ in the equation.\
        gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
    """

    def __init__(self, gamma):
        assert 0 <= gamma <= 1.0, "gamma should be in [0, 1.0]."
        self.gamma = gamma

    def __call__(self, img):
        gamma = np.random.uniform(
            low=max(0, 1 - self.gamma), high=(1 + self.gamma))
        return F.adjust_gamma(img, gamma, gain=1)


class ColorShift(object):
    """
    Args:
        brightness (float) – How much to jitter brightness. brightness_factor is chosen uniformly from \
        [max(0, 1 - brightness), 1 + brightness].
        contrast (float) – How much to jitter contrast. contrast_factor is chosen uniformly from \
        [max(0, 1 - contrast), 1 + contrast].
        saturation (float) – How much to jitter saturation. saturation_factor is chosen uniformly from \
        [max(0, 1 - saturation), 1 + saturation].
        gamma (float) – Non negative real number, same as γ in the equation.\
        gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
    """

    def __init__(self, brightness, contrast, saturation, gamma, isVideo=False):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gamma = gamma
        self.isVideo = isVideo

    def __call__(self, sample):
        color_coms = transforms.Compose([
            BrightShift(self.brightness),
            ContrastShift(self.contrast),
            SaturationShift(self.saturation),
            GammaShift(self.gamma)])

        if self.isVideo:
            src, tar = [], sample['tar']
            for i in sample['src']:
                src.append(color_coms(i))
        else:
            src = color_coms(sample['src'])
            tar = sample['tar']

        return {'src': src,
                'tar': tar}


class RandomCrop(object):
    """Crop randomly the src in a sample.

    Args:
        dsize (sequence or int): Desired output size of the crop. If dsize is an
            int instead of sequence like (h, w), a square crop (dsize, dsize) is
            made.
        skip (sequence or int): Optional skipping on each border
            of the src.
    """

    def __init__(self, dsize, skip=0, isVideo=False):
        assert isinstance(dsize, int) or (isinstance(
            dsize, collections.Iterable) and len(dsize) == 2)
        assert isinstance(dsize, int) or (isinstance(
            dsize, collections.Iterable) and len(dsize) == 2)
        if isinstance(dsize, int):
            self.dsize = (int(dsize), int(dsize))
        else:
            self.dsize = dsize
        if isinstance(skip, int):
            self.skip = (int(skip), int(skip))
        else:
            self.skip = skip
        self.isVideo = isVideo

    def __call__(self, sample):
        src, tar = sample['src'], sample['tar']
        if self.isVideo:
            _, h, w, ch = src.shape
        else:
            h, w, ch = src.shape
        new_h, new_w = self.dsize
        top = random.randint(self.skip[0], h - new_h - self.skip[0])
        left = random.randint(self.skip[1], w - new_w - self.skip[1])
        if self.isVideo:
            src = src[:, top: top + new_h,
                      left: left + new_w]
            tar = tar[:, top: top + new_h,
                      left: left + new_w]
        else:
            src = src[top: top + new_h,
                      left: left + new_w]
            tar = tar[top: top + new_h,
                      left: left + new_w]
        return {'src': src, 'tar': tar}


class CenterCrop(object):
    """Crop randomly the src in a sample.

    Args:
        dsize (sequence or int): Desired output size of the crop. If dsize is an
            int instead of sequence like (h, w), a square crop (dsize, dsize) is
            made.
        skip (sequence or int): Optional skipping on each border
            of the src.
    """

    def __init__(self, dsize, skip=0):
        assert isinstance(dsize, int) or (isinstance(
            dsize, collections.Iterable) and len(dsize) == 2)
        assert isinstance(dsize, int) or (isinstance(
            dsize, collections.Iterable) and len(dsize) == 2)
        if isinstance(dsize, int):
            self.dsize = (int(dsize), int(dsize))
        else:
            self.dsize = dsize
        if isinstance(skip, int):
            self.skip = (int(skip), int(skip))
        else:
            self.skip = skip

    def __call__(self, sample):
        src, tar = sample['src'], sample['tar']
        h, w = src.shape[:2]
        new_h, new_w = self.dsize
        top = (h - 2 * self.skip[0] - new_h) // 2
        left = (w - 2 * self.skip[1] - new_w) // 2
        src = src[top: top + new_h,
                  left: left + new_w]
        tar = tar[top: top + new_h,
                  left: left + new_w]
        return {'src': src, 'tar': tar}


class Resize(object):
    """Resize shorter side to a given value and randomly scale."""

    def __init__(self, dsize, isVideo=False):
        # dsize => [h, w]
        self.dsize = dsize
        # interpolation [1 => nearest, 2 => bilinear]
        self.interpolation = 2
        self.isVideo = isVideo

    def __call__(self, sample):
        if self.isVideo:
            src, tar = [], []
            for s, t in zip(sample['src'], sample['tar']):
                src.append(F.resize(s, self.dsize, self.interpolation))
                tar.append(F.resize(t, self.dsize, self.interpolation)) 
        else:
            src, tar = sample['src'], sample['tar']
            src = F.resize(src, self.dsize, self.interpolation)
            tar = F.resize(tar, self.dsize, self.interpolation)
        return {'src': src, 'tar': tar}


class RandomVerticalFlip(object):
    """Randomly flip the src and the tar"""

    def __init__(self, isVideo=False):
        self.isVideo = isVideo
        pass

    def __call__(self, sample):
        condition = random.randint(0, 1)
        if self.isVideo:
            src, tar = [], []
            for s, t in zip(sample['src'], sample['tar']):
                if condition:
                    s = F.vflip(s)
                    t = F.vflip(t)
                src.append(s)
                tar.append(t) 
        else:
            src, tar = sample['src'], sample['tar']
            if condition:
                src = F.vflip(src)
                tar = F.vflip(tar)
        return {'src': src, 'tar': tar}


class RandomHorizontalFlip(object):
    """Randomly flip the src and the tar"""

    def __init__(self, isVideo=False):
        self.isVideo = isVideo
        pass

    def __call__(self, sample):
        condition = random.randint(0, 1)
        if self.isVideo:
            src, tar = [], []
            for s, t in zip(sample['src'], sample['tar']):
                if condition:
                    s = F.hflip(s)
                    t = F.hflip(t)
                src.append(s)
                tar.append(t) 
        else:
            src, tar = sample['src'], sample['tar']
            if condition:
                src = F.hflip(src)
                tar = F.hflip(tar)
        return {'src': src, 'tar': tar}


class Normalize(object):
    """Normalize a tensor src with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalise each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, sample):
        src = sample['src']
        tar = sample['tar']
        return {'src': (src - self.mean),
                'tar': tar}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        src, tar = sample['src'], sample['tar']
        # swap color axis [HWC]->[CHW]
        src = src.transpose((2, 0, 1))
        tar = tar.transpose((2, 0, 1))
        return {'src': torch.from_numpy(src).float(),
                'tar': torch.from_numpy(tar).float()}


class ToArray(object):
    """Convert PIL Image in sample to ndarray."""
    def __init__(self, isVideo=False):
        self.isVideo = isVideo
        pass

    def __call__(self, sample):
        if self.isVideo:
            src, tar = [], []
            for s, t in zip(sample['src'], sample['tar']):
                src.append(np.expand_dims(np.array(s), 0))
                tar.append(np.expand_dims(np.array(t), 0))
            src = np.concatenate(src, axis=0)
            tar = np.concatenate(tar, axis=0)
        else:
            src = np.array(sample['src'])
            tar = np.array(sample['tar'])
        return {'src': src,
                'tar': tar}
