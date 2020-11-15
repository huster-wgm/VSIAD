#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import re
import torch
from models.fcn import fcn8s, fcn16s, fcn32s
from models.unet import unet
from models.fpn import fpn
from models.resnet import res18net, res34net
from models.resunet import res18unet, res34unet
from models.resunetnb import res18unetNB, res34unetNB
from models.resynet import res18ynet, res34ynet
from models.resynetsync import res18ynetsync, res34ynetsync
from models.resynetsyncnb import res18ynetsyncNB, res34ynetsyncNB


DIR = os.path.dirname(os.path.abspath(__file__))
Checkpoint_DIR = os.path.join(DIR, '../checkpoint')


def load_model(net, src_ch, tar_ch, cuda):
    if net == "res18ynetsyncGN":
        net = eval('res18ynetsync')(src_ch, tar_ch, True, 'GN')
    else:
        net = eval(net)(src_ch, tar_ch, True, 'IN')
    if cuda:
        net.cuda()
    return net


def load_checkpoint(checkpoint, src_ch, tar_ch, cuda):
    assert os.path.exists("{}/{}".format(Checkpoint_DIR, checkpoint)
                          ), "{} not exists.".format(checkpoint)
    print("Loading checkpoint: {}".format(checkpoint))
    net = checkpoint.split('-')[0]
    if "@" in net:
        net = net.split('@')[0]
    if net == "res18ynetsyncGN":
        net = eval('res18ynetsync')(src_ch, tar_ch, False, 'GN')
    else:
        net = eval(net)(src_ch, tar_ch, False, 'IN')
    net.load_state_dict(torch.load(os.path.join(Checkpoint_DIR, checkpoint),
                                   map_location=lambda storage, loc: storage))
    if cuda:
        net.cuda()
    return net.eval()

def natural_sort(unsorted_list):
    # refer to  https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(unsorted_list, key=alphanum_key)


def pair_validate(pFiles, gFiles):
    valid = True
    if len(pFiles) == len(gFiles):
        for pfile, gfile in zip(pFiles, gFiles):
            if os.path.basename(pfile) != os.path.basename(gfile):
                valid = False
                # print('{} and {} isn\'t consistent.'.format(os.path.basename(pfile), os.path.basename(gfile)))
    else:
        valid = False
        # find different
        xlen = min(len(pFiles), len(gFiles))
        for pfile, gfile in zip(pFiles[:xlen], gFiles[:xlen]):
            if os.path.basename(pfile) != os.path.basename(gfile):            
                print('{} and {} isn\'t consistent.'.format(os.path.basename(pfile), os.path.basename(gfile)))
        print("Warning >> Extra P:\n", len(pFiles[xlen:]))
        print("Warning >> Extra G:\n", len(gFiles[xlen:]))
    return valid
