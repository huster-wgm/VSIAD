#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imsave
from skimage.color import lab2rgb, rgb2lab, rgb2gray

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transform import *

import warnings
warnings.filterwarnings("ignore")

Src_DIR = os.path.dirname(os.path.abspath(__file__))
Dataset_DIR = os.path.join(Src_DIR, '../datasets/')


def showPixelRange(arr, channels=['R','G','B']):
    """
    args:
        arr: 3d img array
        channels: list of channel name
    return 0
    """
    assert arr.shape[-1] == len(channels), "Channel should be consistent."
    print("{} Image:".format(''.join(channels)))
    for idx, ch in enumerate(channels):
        ch_max, ch_min = np.max(arr[:,:,idx]), np.min(arr[:,:,idx])
        print("\t{}-Channel : Max:{} ; Min:{} ;".format(ch, ch_max, ch_min))


class Basic(Dataset):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.split = split
        # file = '{}-nir.txt'.format(self.split)
        file = '{}.txt'.format(self.split)
        with open(os.path.join(Dataset_DIR, self.root, file), 'r') as f:
            self.datalist = [line.strip() for line in f.readlines()]

        self.srcpath = os.path.join(Dataset_DIR, root, "src", '%s')
        self.tarpath = os.path.join(Dataset_DIR, root, "tar", '%s')
        
        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    @staticmethod
    def normalize(arr):
        mx = np.max(arr)
        mi = np.min(arr)
        arr = (arr - mi) / (mx - mi)
        return arr

    def _whitespace(self, img, width=5):
        """
        Args:
            img : ndarray [h,w,c]
        """
        row, col, ch = img.shape
        tmp = np.ones((row + 2*width, col + 2*width, ch), "uint8") * 255
        tmp[width:row+width,width:width+col,:] = img
        return tmp
    
    def _g2img(self, arr, whitespace=True):
        """
        Args:
            arr (str): ndarray [h,w,c]
        """
        if arr.shape[-1] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        img = (arr * 255).astype("uint8")
        if whitespace:
            img = self._whitespace(img)
        return img


    def _rgb2img(self, arr, whitespace=True):
        """
        Args:
            arr (str): ndarray [h,w,c]
        """
        if arr.shape[-1] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        img = (arr * 255).astype("uint8")
        if whitespace:
            img = self._whitespace(img)
        return img

    def _lab2img(self, lab, whitespace=True):
        """
        Args:
            lab: LAB in [0, 1.0]
        """
        lab[:,:,:1] = lab[:,:,:1] * 100
        lab[:,:,1:] = lab[:,:,1:] * 255 - 128
        img = (lab2rgb(lab.astype("float64")) * 255).astype("uint8")
        if whitespace:
            img = self._whitespace(img)
        return img

    def _arr2gray(self, arr):
        """
        Args:
            arr: ndarray
        return tensor(L)
        """
        arr = rgb2gray(arr)
        arr = np.expand_dims(arr, axis=-1).transpose((2, 0, 1))
        tensor = torch.from_numpy(arr).float()
        return tensor

    def _arr2rgb(self, arr):
        """
        Args:
            arr: ndarray
        return tensor(RGB)
        """
        arr = arr / (2**8-1)
        arr = arr.transpose((2, 0, 1))
        tensor = torch.from_numpy(arr).float()
        return tensor

    def _arr2lab(self, arr):
        """
        Args:
            arr: ndarray
        return tensor(LAB)
        """
        arr = rgb2lab(arr)
        arr[:,:,:1] = arr[:,:,:1] / 100
        arr[:,:,1:] = (arr[:,:,1:] + 128) / 255
        arr = arr.transpose((2, 0, 1))
        tensor = torch.from_numpy(arr).float()
        return tensor

    def _arr2rgb3D(self, arr):
        """
        Args:
            arr: ndarray [frames, row, col, ch]
        return tensor(RGB) [ch, frames, row, col]
        """
        arr = arr / (2**8-1)
        arr = arr.transpose((3, 0, 1, 2))
        tensor = torch.from_numpy(arr).float()
        return tensor

    def _arr2lab3D(self, arr):
        """
        Args:
            arr: ndarray [frames, row, col, ch]
        return tensor(LAB) [ch, frames, row, col]
        """
        new_arr = []
        frames, row, col, ch = arr.shape
        for f in range(frames):
            arrf = rgb2lab(arr[f,:,:,:])
            arrf[:,:,:1] = arrf[:,:,:1] / 100
            arrf[:,:,1:] = (arrf[:,:,1:] + 128) / 255
            new_arr.append(np.expand_dims(arrf, axis=0))
        new_arr = np.concatenate(new_arr, axis=0).transpose((3, 0, 1, 2))
        tensor = torch.from_numpy(new_arr).float()
        return tensor


class G2G(Basic):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 1
        self.tar_ch = 1
        self.ver = 'G2G'

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        # src => arr => tensor(L)
        src = self._arr2gray(sample['src'])
        # tar => arr => tensor(RGB)
        tar = self._arr2gray(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        # showPixelRange(src, channels=['R', 'G', 'B'])
        # showPixelRange(tar, channels=['R', 'G', 'B'])
        # convert bayer array to RGB
        nir_rgb = self._rgb2img(src)
        rgb_rgb = self._rgb2img(tar)

        vis_img = np.concatenate([nir_rgb, rgb_rgb], axis=1)
        save_dir = os.path.join(Src_DIR, "../example", self.root+self.ver)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}.png".format(save_dir, self.split, idx), vis_img)


class RGB2G(G2G):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 3
        self.tar_ch = 1
        self.ver = 'RGB2G'

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        # src => arr => tensor(L)
        src = self._arr2rgb(sample['src'])
        # tar => arr => tensor(RGB)
        tar = self._arr2gray(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample


class L2RGB(Basic):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 1
        self.tar_ch = 3
        self.ver = 'L2RGB'

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        # src => arr => tensor(L)
        src = self._arr2gray(sample['src'])
        # tar => arr => tensor(RGB)
        tar = self._arr2rgb(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        # showPixelRange(src, channels=['R', 'G', 'B'])
        # showPixelRange(tar, channels=['R', 'G', 'B'])
        # convert bayer array to RGB
        nir_rgb = self._rgb2img(src)
        rgb_rgb = self._rgb2img(tar)

        vis_img = np.concatenate([nir_rgb, rgb_rgb], axis=1)
        save_dir = os.path.join(Src_DIR, "../example", self.root+self.ver)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}.png".format(save_dir, self.split, idx), vis_img)


class LN2RGB(L2RGB):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 1
        self.tar_ch = 3
        self.ver = 'LN2RGB'

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        sample['src'] = self.normalize(sample['src'])
        # src => arr => tensor(L)
        src = self._arr2gray(sample['src'])
        # tar => arr => tensor(RGB)
        tar = self._arr2rgb(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample



class G2RGB(L2RGB):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 1
        self.tar_ch = 3
        self.ver = 'G2RGB'

    def __getitem__(self, idx):
        
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        # src => arr => tensor(L)
        src = self._arr2gray(sample['tar'])
        # tar => arr => tensor(RGB)
        tar = self._arr2rgb(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample



class L2LAB(Basic):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 1
        self.tar_ch = [1, 2]
        self.ver = 'L2LAB'

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        # src => arr => tensor(L)
        src = self._arr2gray(sample['src'])
        # tar => arr => tensor(LAB)
        tar = self._arr2lab(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        # showPixelRange(src, channels=['NIR'])
        # showPixelRange(tar, channels=['A', 'B'])
        # convert array to RGB img
        nir_img = self._rgb2img(src)
        nirab_img = self._lab2img(tar)

        vis_img = np.concatenate([nir_img, nirab_img], axis=1)
        save_dir = os.path.join(Src_DIR, "../example", self.root+self.ver)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}.png".format(save_dir, self.split, idx), vis_img)

        
class LN2LAB(L2LAB):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 1
        self.tar_ch = [1, 2]
        self.ver = 'LN2LAB'

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        sample['src'] = self.normalize(sample['src'])
        # src => arr => tensor(L)
        src = self._arr2gray(sample['src'])
        # tar => arr => tensor(LAB)
        tar = self._arr2lab(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample


class G2LAB(L2LAB):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 1
        self.tar_ch = [1, 2]
        self.ver = 'G2LAB'

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        sample['src'] = self.normalize(sample['src'])
        # src => arr => tensor(L)
        src = self._arr2gray(sample['tar'])
        # tar => arr => tensor(LAB)
        tar = self._arr2lab(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample


class RGB2RGB(Basic):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 3
        self.tar_ch = 3
        self.ver = 'RGB2RGB'

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        # src => arr => tensor(RGB)
        src = self._arr2rgb(sample['src'])
        # tar => arr => tensor(RGB)
        tar = self._arr2rgb(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        # showPixelRange(src, channels=['R', 'G', 'B'])
        # showPixelRange(tar, channels=['R', 'G', 'B'])
        # convert bayer array to RGB
        nir_rgb = self._rgb2img(src)
        rgb_rgb = self._rgb2img(tar)

        vis_img = np.concatenate([nir_rgb, rgb_rgb], axis=1)
        save_dir = os.path.join(Src_DIR, "../example", self.root+self.ver)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}.png".format(save_dir, self.split, idx), vis_img)


class RGB2LAB(Basic):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 3
        self.tar_ch = [1, 2]
        self.ver = 'RGB2LAB'

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        # src >> arr => tensor(RGB)
        src = self._arr2rgb(sample['src'])
        # tar >> arr => tensor(LAB)
        tar = self._arr2lab(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        # showPixelRange(src, channels=['R', 'G', 'B'])
        # showPixelRange(tar, channels=['L', 'A', 'B'])
        # convert bayer array to RGB
        nir_rgb = self._rgb2img(src)
        rgb_rgb = self._lab2img(tar)

        vis_img = np.concatenate([nir_rgb, rgb_rgb], axis=1)
        save_dir = os.path.join(Src_DIR, "../example", self.root+self.ver)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}.png".format(save_dir, self.split, idx), vis_img)



class LAB2LAB(Basic):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 3
        self.tar_ch = [1, 2]
        self.ver = 'LAB2LAB'

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        # src >> arr => tensor(LAB)
        src = self._arr2lab(sample['src'])
        # tar >> arr => tensor(LAB)
        tar = self._arr2lab(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        # showPixelRange(src, channels=['L', 'A', 'B'])
        # showPixelRange(tar, channels=['L', 'A', 'B'])
        # convert bayer array to RGB
        nir_rgb = self._lab2img(src)
        rgb_rgb = self._lab2img(tar)

        vis_img = np.concatenate([nir_rgb, rgb_rgb], axis=1)
        save_dir = os.path.join(Src_DIR, "../example", self.root+self.ver)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}.png".format(save_dir, self.split, idx), vis_img)


class GAB2LAB(LAB2LAB):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 3
        self.tar_ch = [1, 2]
        self.ver = 'GAB2LAB'

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        # src >> arr => tensor(LAB)
        gray = self._arr2gray(sample['tar'])
        src = self._arr2lab(sample['src'])
        src[0,:,:] = gray[0]
        # tar >> arr => tensor(LAB)
        tar = self._arr2lab(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        # showPixelRange(src, channels=['L', 'A', 'B'])
        # showPixelRange(tar, channels=['L', 'A', 'B'])
        # convert bayer array to RGB
        nir_rgb = self._lab2img(src)
        rgb_rgb = self._lab2img(tar)

        vis_img = np.concatenate([nir_rgb, rgb_rgb], axis=1)
        save_dir = os.path.join(Src_DIR, "../example", self.root+self.ver)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}.png".format(save_dir, self.split, idx), vis_img)


class LAB2RGB(Basic):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 3
        self.tar_ch = [1, 2]
        self.ver = 'LAB2RGB'

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file)
        tar = Image.open(tar_file)
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        # src >> arr => tensor(LAB)
        src = self._arr2lab(sample['src'])
        # tar >> arr => tensor(RGB)
        tar = self._arr2rgb(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        # showPixelRange(src, channels=['L', 'A', 'B'])
        # showPixelRange(tar, channels=['L', 'A', 'B'])
        # convert bayer array to RGB
        nir_rgb = self._lab2img(src)
        rgb_rgb = self._rgb2img(tar)

        vis_img = np.concatenate([nir_rgb, rgb_rgb], axis=1)
        save_dir = os.path.join(Src_DIR, "../example", self.root+self.ver)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}.png".format(save_dir, self.split, idx), vis_img)


class RGB2LAB3D(Basic):
    def __init__(
            self, root, split='all', transform=None, frames=2):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)
        files = self.datalist
        self.datalist = []
        if self.split == "train":
            for i in range(0, len(files), frames//2):
                fset = files[i:i+frames]
                if len(fset) == frames: 
                    self.datalist.append(fset)
        else:
            for i in range(0, len(files), frames):
                self.datalist.append(files[i:i+frames])

        self.src_ch = 3
        self.tar_ch = [1, 2]
        self.ver = 'RGB2LAB3D'

    def __getitem__(self, idx):
        src, tar = [], []
        for file in self.datalist[idx]:
            src_file = self.srcpath % file
            tar_file = self.tarpath % file

            src.append(Image.open(src_file))
            tar.append(Image.open(tar_file))
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            src, tar = [], []
            for s, t in zip(sample['src'], sample['tar']):
                src.append(np.expand_dims(np.array(s), 0))
                tar.append(np.expand_dims(np.array(t), 0))
            src = np.concatenate(src, axis=0)
            tar = np.concatenate(tar, axis=0)
        # src >> arr => tensor(RGB)
        src = self._arr2rgb3D(sample['src'])
        # tar >> arr => tensor(LAB)
        tar = self._arr2lab3D(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        # ch, frames, row, col
        src = sample['src'].numpy()[:,0,:,:].transpose((1, 2, 0))
        tar = sample['tar'].numpy()[:,0,:,:].transpose((1, 2, 0))
        # showPixelRange(src, channels=['R', 'G', 'B'])
        # showPixelRange(tar, channels=['L', 'A', 'B'])
        # convert bayer array to RGB
        nir_rgb = self._rgb2img(src)
        rgb_rgb = self._lab2img(tar)

        vis_img = np.concatenate([nir_rgb, rgb_rgb], axis=1)
        save_dir = os.path.join(Src_DIR, "../example", self.root+self.ver)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}.png".format(save_dir, self.split, idx), vis_img)


def load_dataset(root, version, mode="training"):
    """
    Args:
        root (str): root of dataset
        version (str): version of dataset
        mode (str): ['training', 'evaluating']
    """
    # ori[h, w] = [900, 1200]
    # trsize = (600, 800)
    # dsize = (448, 448)
    trsize = (300, 400)
    dsize = (256, 256)
    vrsize = (480, 640)
    if '3D' in version:
        isVideo = True
    else:
        isVideo = False
    # setup transform
    if mode == "training":
        t_trans = transforms.Compose([
            ColorShift(0.1, 0.1, 0.1, 0.1, isVideo=isVideo),
            RandomHorizontalFlip(isVideo=isVideo),
            RandomVerticalFlip(isVideo=isVideo),
            Resize(trsize, isVideo=isVideo),
            ToArray(isVideo=isVideo),
            RandomCrop(dsize, isVideo=isVideo),
            ])
        v_trans = transforms.Compose([
            Resize(trsize, isVideo=isVideo),
            ToArray(isVideo=isVideo),
            RandomCrop(dsize, isVideo=isVideo),
            ])
    else:
        v_trans = transforms.Compose([
            Resize(vrsize, isVideo=isVideo),
            ToArray(isVideo=isVideo),
            ])
        t_trans = v_trans

    # setup dataset
    if "3D" in version:
        if 'x' in version:
            version, frames = version.split('x')
        else:
            frames = 4
        trainset = eval(version)(root=root, split="train", transform=t_trans, frames=int(frames))
        valset = eval(version)(root=root, split="val", transform=v_trans, frames=int(frames))
        testset = eval(version)(root=root, split="test", transform=v_trans, frames=int(frames))
    else:
        trainset = eval(version)(root=root, split="train", transform=t_trans)
        valset = eval(version)(root=root, split="val", transform=v_trans)
        testset = eval(version)(root=root, split="test", transform=v_trans)
    return trainset, valset, testset



if __name__ == "__main__":
    # setup parameters
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-idx', type=int, default=0,
                        help='index of sample image')
    args = parser.parse_args()
    idx = args.idx
    for root in ['VC24']:
        for ver in ["RGB2LAB", "RGB2LAB3D"]:
            for stage in ["training", "testing"]:
                trainset, valset, testset = load_dataset(root, ver, stage)
                # print("Load train set = {} examples, val set = {} examples".format(
                #     len(trainset), len(valset)))
                sample = trainset[idx]
                trainset.show(idx)
                valset.show(idx)
                testset.show(idx)
                print("Tensor size of {}/{}/{}".format(root, ver, stage))
                print("\tsrc:", sample["src"].shape,
                      "tar:", sample["tar"].shape,)
