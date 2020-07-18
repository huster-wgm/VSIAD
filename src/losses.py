#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import numpy as np
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import metrics

eps = 1e-6


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def __repr__(self):
        return "L1"

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss

class L1Loss3D(nn.Module):
    def __init__(self):
        super(L1Loss3D, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def __repr__(self):
        return "L13D"

    def forward(self, output, target):
        nb, ch, frame, row, col = output.shape
        loss = []
        for f in range(frame):
            loss.append(self.criterion(output[:,:,f,:,:], target[:,:,f,:,:]))
        return sum(loss) / len(loss)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
    
    def __repr__(self):
        return "MSE"

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss


class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)

    def __repr__(self):
        return "PSNR"

    def forward(self, output, target):
        mse = self.criterion(output, target)
        loss = 10 * torch.log10(1.0 / mse)
        return loss


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.criterionBinary = nn.BCELoss(size_average=True)
        self.criterionMulti = nn.NLLLoss(size_average=True)

    def __repr__(self):
        return "CE"

    def forward(self, output, target):
        if target.shape[1] == 1:
            # binary cross enthropy
            loss = self.criterionBinary(output, target)
        else:
            # multi-class cross enthropy
            target = torch.argmax(target, dim=1).long()
            loss = self.criterionMulti(torch.log(output), target)
        return loss


class DSSIMLoss(nn.Module):
    def __init__(self):
        super(DSSIMLoss, self).__init__()
        self.criterion = metrics.SSIM()

    def __repr__(self):
        return "DSSIM"

    def forward(self, output, target):
        loss = (1. - self.criterion(output, target)) / 2.
        return loss


class DSSIMLoss3D(nn.Module):
    def __init__(self):
        super(DSSIMLoss3D, self).__init__()
        self.criterion = metrics.SSIM()

    def __repr__(self):
        return "DSSIM3D"

    def forward(self, output, target):
        nb, ch, frame, row, col = output.shape
        loss = []
        for f in range(frame):
            loss.append((1. - self.criterion(output[:,:,f,:,:], target[:,:,f,:,:])) / 2.)
        return sum(loss) / len(loss)


class NearestSelector(object):
    def __init__(self, shift=2, stride=1, criter='l1'):
        self.shift = shift
        self.stride = stride
        self.criter = criter

    def __repr__(self):
        return "NS"

    @staticmethod
    def unravel_index(tensor, cols):
        """
        args:
            tensor : 2D tensor, [nb, rows*cols]
            cols : int
        return 2D tensor nb * [rowIndex, colIndex]
        """
        index = torch.argmin(tensor, dim=1).view(-1,1)
        rIndex = index / cols
        cIndex = index % cols
        minRC = torch.cat([rIndex, cIndex], dim=1)
        # print("minRC", minRC.shape, minRC)
        return minRC

    def shift_diff(self, output, target, crop_row, crop_col):
        diff = []
        for i in range(0, 2 * self.shift):
            for j in range(0, 2 * self.shift):
                output_crop = output[:, :, 
                                     self.shift * self.stride: self.shift * self.stride + crop_row,
                                     self.shift * self.stride: self.shift * self.stride + crop_col,]
                target_crop = target[:, :, 
                                     i * self.stride: i * self.stride + crop_row,
                                     j * self.stride: j * self.stride + crop_col,]
                diff_ij = torch.sum(abs(target_crop-output_crop), dim=[1,2,3]).view(-1,1)
                diff.append(diff_ij)
        return torch.cat(diff, dim=1)
        
    def crop(self, output, target):
        nb, ch, row, col = output.shape
        crop_row = row - 2 * self.shift * self.stride
        crop_col = col - 2 * self.shift * self.stride
        diff = self.shift_diff(output.detach(), target.detach(), crop_row, crop_col)
        minRC = self.unravel_index(diff, 2 * self.shift)
        crop = [self.shift * self.stride, self.shift * self.stride + crop_row,
                self.shift * self.stride, self.shift * self.stride + crop_col]
        output_ = output[:,
                         :,
                        crop[0] : crop[1],
                        crop[2] : crop[3]]
        target_ = torch.zeros(*output_.shape).to(target.device)
        for idx, (minR, minC) in enumerate(minRC):
            target_[idx] = target[idx,
                                  :,
                                  minR * self.stride: minR * self.stride + crop_row,
                                  minC * self.stride: minC * self.stride + crop_row]
        return output_, target_

    
class ConLoss(nn.Module):
    """
    Consistency of samples within batch
    """

    def __init__(self):
        super(ConLoss, self).__init__()
        self.criterMSE = nn.MSELoss(size_average=True)

    def __repr__(self):
        return 'ConLoss'

    def forward(self, feats):
        feat_max, _ = torch.max(feats, dim=0)
        feat_min, _ = torch.min(feats, dim=0)
        zeros = torch.zeros(feat_max.shape).to(feats.device)
        return self.criterMSE(torch.abs(feat_max - feat_min), zeros)


class CrossLoss(nn.Module):
    """
    Cross comparison between samples within batch
    """

    def __init__(self):
        super(CrossLoss, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def __repr__(self):
        return 'CrossLoss'

    def forward(self, output, target):
        nb, ch, row, col = output.shape
        output = output[:nb-1, :, :, :]
        target = target[1:nb, :, :, :]
        return self.criterion(output, target)


class FLoss(nn.Module):
    """
    Focal Loss
    Lin, Tsung-Yi, et al. \
    "Focal loss for dense object detection." \
    Proceedings of the IEEE international conference on computer vision. 2017.
    (modified from https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py)
    """
    def __init__(self, gamma=2., weight=None, size_average=True):
        super(FLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def __repr__(self):
        return 'Focal'

    def _get_weights(self, y_true, nb_ch):
        """
        args:
            y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
            nb_ch : int 
        return [float] weights
        """
        batch_size, img_rows, img_cols = y_true.shape
        pixels = batch_size * img_rows * img_cols
        weights = [torch.sum(y_true==ch).item() / pixels for ch in range(nb_ch)]
        return weights

    def forward(self, output, target):
        output = torch.clamp(output, min=eps, max=(1. - eps))
        if target.shape[1] == 1:
            # binary focal loss
            # weights = self._get_weigthts(target[:,0,:,:], 2)
            alpha = 0.1
            loss = - (1.-alpha) * ((1.-output)**self.gamma)*(target*torch.log(output)) \
              - alpha * (output**self.gamma)*((1.-target)*torch.log(1.-output))
        else:
            # multi-class focal loss
            # weights = self._get_weigthts(torch.argmax(target, dim=1), target.shape[1])
            loss = - ((1.-output)**self.gamma)*(target*torch.log(output))

        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()
        
        
class VGG16Loss(nn.Module):
    def __init__(self, requires_grad=False, cuda=True):
        super(VGG16Loss, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if cuda:
            self.slice1.cuda()
            self.slice2.cuda()
            self.slice3.cuda()
            self.slice4.cuda()

    def __repr__(self):
        return "VGG16"

    def forward(self, output, target):
        nb, ch, row, col = output.shape
        if ch == 1:
            output = torch.cat([output, output, output], dim=1)
            target = torch.cat([target, target, target], dim=1)
        ho = self.slice1(output)
        ht = self.slice1(target)
        h_relu1_2_loss = self.criterion(ho,ht)
        ho = self.slice2(ho)
        ht = self.slice2(ht)
        h_relu2_2_loss = self.criterion(ho,ht)
        ho = self.slice3(ho)
        ht = self.slice3(ht)
        h_relu3_3_loss = self.criterion(ho,ht)
        ho = self.slice4(ho)
        ht = self.slice4(ht)
        h_relu4_3_loss = self.criterion(ho,ht)
        return sum([h_relu1_2_loss, h_relu2_2_loss, h_relu3_3_loss, h_relu4_3_loss]) / 4


class VGG16Loss3D(nn.Module):
    def __init__(self, requires_grad=False, cuda=True):
        super(VGG16Loss3D, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if cuda:
            self.slice1.cuda()
            self.slice2.cuda()
            self.slice3.cuda()
            self.slice4.cuda()

    def __repr__(self):
        return "VGG163D"

    def forward2d(self, output, target):
        nb, ch, row, col = output.shape
        if ch == 1:
            output = torch.cat([output, output, output], dim=1)
            target = torch.cat([target, target, target], dim=1)
        ho = self.slice1(output)
        ht = self.slice1(target)
        h_relu1_2_loss = self.criterion(ho,ht)
        ho = self.slice2(ho)
        ht = self.slice2(ht)
        h_relu2_2_loss = self.criterion(ho,ht)
        ho = self.slice3(ho)
        ht = self.slice3(ht)
        h_relu3_3_loss = self.criterion(ho,ht)
        ho = self.slice4(ho)
        ht = self.slice4(ht)
        h_relu4_3_loss = self.criterion(ho,ht)
        return sum([h_relu1_2_loss, h_relu2_2_loss, h_relu3_3_loss, h_relu4_3_loss]) / 4

    def forward(self, output, target):
        nb, ch, frame, row, col = output.shape
        loss = []
        for f in range(frame):
            loss.append(
                self.forward2d(output[:,:,f,:,:], target[:,:,f,:,:]))
        return sum(loss) / len(loss)


if __name__ == "__main__":
    for ch in [3, 1]:
        for cuda in [True, False]:
            batch_size, img_row, img_col = 32, 24, 24
            y_true = torch.rand(batch_size, ch, img_row, img_col)
            y_pred = torch.rand(batch_size, ch, img_row, img_col)
            if cuda:
                y_pred = y_pred.cuda()
                y_true = y_true.cuda()

            print('#'*20, 'Test on cuda : {} ; size : {}'.format(cuda, y_true.size()))

            y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
            criterion = L1Loss()
            print('\t gradient bef : {}'.format(y_pred_.grad))
            loss = criterion(y_pred_, y_true_)
            loss.backward()
            print('\t gradient aft : {}'.format(y_pred_.grad.shape))
            print('{} : {}'.format(repr(criterion), loss.item()))

#             y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
#             criterion = CELoss()
#             loss = criterion(y_pred_, y_true_)
#             loss.backward()
#             print('{} : {}'.format(repr(criterion), loss.item()))
#             print('\t gradient : {}'.format(y_pred_.grad.shape))

#             y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
#             criterion = MSELoss()
#             loss = criterion(y_pred_, y_true_)
#             loss.backward()
#             print('{} : {}'.format(repr(criterion), loss.item()))
#             print('\t gradient : {}'.format(y_pred_.grad.shape))

#             y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
#             criterion = FLoss()
#             loss = criterion(y_pred_, y_true_)
#             loss.backward()
#             print('{} : {}'.format(repr(criterion), loss.item()))
#             print('\t gradient : {}'.format(y_pred_.grad.shape))

#             y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
#             criterion = VGG16Loss(cuda=cuda)
#             loss = criterion(y_pred_, y_true_)
#             loss.backward()
#             print('{} : {}'.format(repr(criterion), loss.item()))
#             print('\t gradient : {}'.format(y_pred_.grad.shape))

            y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
            selector = NearestSelector()
            y_pred_near, y_true_near = selector.crop(y_pred_, y_true_)
            criterion = L1Loss()
            print('\t gradient bef : {}'.format(y_pred_.grad))
            loss = criterion(y_pred_near, y_true_near)
            loss.backward()
            print('\t gradient aft : {}'.format(y_pred_.grad.shape))
            print('{}-near : {}'.format(repr(criterion), loss.item()))

