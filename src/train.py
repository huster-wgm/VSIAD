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
import torch
import torch.optim as optim

from runner import Trainer, yTrainer, yTrainer3D, bergTrainer
from datasets import load_dataset
from utils import load_model


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)



def main(args):
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")

    # initialize datasets
    trainset, valset, _ = load_dataset(args.root, args.ver, "training")
    print("Training with {}-Dataset".format(args.ver))
    print("Load train set = {} examples, val set = {} examples".format(
        len(trainset), len(valset)))

    # load model
    args.src_ch = trainset.src_ch
    args.tar_ch = trainset.tar_ch
    net = load_model(args.net, args.src_ch, args.tar_ch, args.cuda)
    net.optimizer = optim.Adam(
        net.parameters(), lr=args.lr, betas=args.optim_betas)

    # initialize network
    method = "{}-{}-{}".format(args.net, args.root, args.ver)  
    print("Loading method:", method)

    # initialize runner
    print("Start training {}...".format(method))
    if 'ynet' in args.net:
        run = yTrainer(args, method, True, criterion=args.loss)
        if '3D' in args.net:
            run = yTrainer3D(args, method, True, criterion=args.loss)
    elif 'berg' in args.net:
        run = bergTrainer(args, method, True, criterion=args.loss)
    else:
        run = Trainer(args, method, False, criterion=args.loss)
    run.training(net, [trainset, valset])
    run.save_log()
    run.learning_curve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-root', type=str, default='Seq3RGB',
                        help='root dir of dataset')
    parser.add_argument('-ver', type=str, default='LAB2LAB',
                        help='version of dataset')
    parser.add_argument('-net', type=str, default='res18unet',
                        help='network type for training')
    parser.add_argument('-loss', type=str, default='L1Loss',
                        help='loss function for training')
    parser.add_argument('-trigger', type=str, default='epoch', choices=['epoch', 'iter'],
                        help='trigger type for logging')
    parser.add_argument('-interval', type=int, default=10,
                        help='interval for logging')
    parser.add_argument('-terminal', type=int, default=100,
                        help='terminal for training ')
    parser.add_argument('-alpha', type=float, default=10.0,
                        help='weight for perceptual loss(alpha) ')
    parser.add_argument('-batch_size', type=int, default=12,
                        help='batch_size for training ')
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='learning rate for discriminator')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()
    args.optim_betas = (0.9, 0.999)
    main(args)
