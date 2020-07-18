#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import time
import shutil
import metrics
import losses
import numpy as np
import pandas as pd
import warnings

import torch
from torch.utils.data import DataLoader

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

Src_DIR = os.path.dirname(os.path.abspath(__file__))
Logs_DIR = os.path.join(Src_DIR, '../logs')
Checkpoint_DIR = os.path.join(Src_DIR, '../checkpoint')

if not os.path.exists(Logs_DIR):
    os.mkdir(Logs_DIR)
    os.mkdir(os.path.join(Logs_DIR, 'raw'))
    os.mkdir(os.path.join(Logs_DIR, 'curve'))
    os.mkdir(os.path.join(Logs_DIR, 'snapshot'))

if not os.path.exists(Checkpoint_DIR):
    os.mkdir(Checkpoint_DIR)


class Base(object):
    def __init__(self, args, method, is_multi=False, criterion='L1Loss', metric='PSNR'):
        self.args = args
        self.method = method
        self.is_multi = is_multi
        self.date = time.strftime("%h%d_%H")
        if int(args.alpha) == 0:
            self.method += '-nop'
        self.repr = "{}_{}_{}".format(
            self.method, self.args.trigger, self.args.terminal)
        self.epoch = 0
        self.iter = 0
        self.logs = []
        self.criterion = eval("{}.{}()".format('losses', criterion))
        self.evaluator = eval("{}.{}()".format('metrics', metric))
        self.snapshot = os.path.join(Logs_DIR, "snapshot", self.method)
        if not os.path.exists(self.snapshot):
            os.makedirs(self.snapshot)
        else:
            shutil.rmtree(self.snapshot)
            os.makedirs(self.snapshot)
        
        self.header = ["epoch", "iter"]
        for stage in ['trn', 'val']:
            for key in [repr(self.criterion),repr(self.evaluator),"FPS"]:
                self.header.append("{}_{}".format(stage, key))

    def logging(self, verbose=True):
        self.logs.append([self.epoch, self.iter] +
                         self.trn_log + self.val_log)
        if verbose:
            str_a = ['{}:{:05d}'.format(k,v) for k,v in zip(self.header[:2], [self.epoch, self.iter])]
            str_b = ['{}:{:.2f}'.format(k,v) for k,v in zip(self.header[2:], self.trn_log + self.val_log)]
            print(', '.join(str_a + str_b))

    def save_log(self):
        self.logs = pd.DataFrame(self.logs,
                                 columns=self.header)
        self.logs.to_csv(os.path.join(Logs_DIR, 'raw', '{}.csv'.format(self.repr)), index=False, float_format='%.3f')

        speed_info = [self.repr, self.logs.iloc[:, 4].mean(), self.logs.iloc[:, 7].mean()]
        df = pd.DataFrame([speed_info],
                          columns=["experiment", self.header[4], self.header[7]])
        if os.path.exists(os.path.join(Logs_DIR, 'speed.csv')):
            prev_df = pd.read_csv(os.path.join(Logs_DIR, 'speed.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Logs_DIR, 'speed.csv'), index=False, float_format='%.3f')

    def save_checkpoint(self, net):
        torch.save(net.state_dict(), os.path.join(Checkpoint_DIR, "{}.pth".format(self.repr)))

    def learning_curve(self, idxs=[2,3,5,6]):
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        # set style
        sns.set_context("paper", font_scale=1.5,)
        # sns.set_style("ticks", {
        #     "font.family": "Times New Roman",
        #     "font.serif": ["Times", "Palatino", "serif"]})

        for idx in idxs:
            plt.plot(self.logs[self.args.trigger],
                     self.logs[self.header[idx]], label=self.header[idx])
        plt.ylabel(" {} / {} ".format(repr(self.criterion), repr(self.evaluator)))
        if self.args.trigger == 'epoch':
            plt.xlabel("Epochs")
        else:
            plt.xlabel("Iterations")
        plt.suptitle("Training log of {}".format(self.method))
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.savefig(os.path.join(Logs_DIR, 'curve', '{}.png'.format(self.repr)),
                    format='png', bbox_inches='tight', dpi=144)

    def save_snapshot(self, src, tar, gen, dataset):
        """
          Args:
            src: (tensor) tensor of src
            tar: (tensor) tensor of tar
            gen: (tensor) tensor of prediction
        """
        import random
        from skimage.io import imsave

        # transfer to cpu
        idx = 0
        if self.args.cuda:
            src = src.cpu()
            tar = tar.cpu()
            gen = gen.cpu()
        if '3D' in dataset.ver:
            src = src.numpy()[idx, :, idx].transpose((1, 2, 0))
            tar = tar.numpy()[idx, :, idx].transpose((1, 2, 0))
            gen = gen.numpy()[idx, :, idx].transpose((1, 2, 0))
        else:
            src = src.numpy()[idx].transpose((1, 2, 0))
            tar = tar.numpy()[idx].transpose((1, 2, 0))
            gen = gen.numpy()[idx].transpose((1, 2, 0))
        src_type, tar_type = dataset.ver.split("2")
        if "LAB" in src_type:
            src_img = dataset._lab2img(src)
        else:
            src_img = dataset._rgb2img(src)
        if "LAB" != tar_type:
            tar_img = dataset._rgb2img(tar)
            gen_img = dataset._rgb2img(gen)
        else:
            tar_img = dataset._lab2img(tar)
            gen_img = dataset._lab2img(gen)
        vis_img = np.concatenate([src_img, tar_img, gen_img], axis=0)
        # save image
        imsave(os.path.join(self.snapshot, '{}_iter-{:05d}.png'.format(self.method, self.iter)), vis_img)


class Trainer(Base):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True,)
            for idx, sample in enumerate(data_loader):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_y = net(x)
                loss = self.criterion(gen_y, y)
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3), 
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]
 
                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=True, pin_memory=True,)
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
               # forwading
                gen_y = net(x)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3), 
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)


class yTrainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        criterionExtra = losses.VGG16Loss()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True,)
            for idx, sample in enumerate(data_loader):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_ab], dim=1)
                # loss_l = self.criterion(gen_y[:,:1,:,:], y[:,:1,:,:])
                # loss_ab = self.criterion(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss_l = criterionA(gen_y[:,:1,:,:], y[:,:1,:,:])
                loss_ab = criterionB(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss_lab = criterionExtra(gen_y, y)
                loss = loss_l + loss_ab + args.alpha * loss_lab
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3), 
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]
 
                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False,  float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True,)
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_ab], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3), 
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)


class yTrainer3D(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()
        criterionA = losses.DSSIMLoss3D()
        criterionB = losses.L1Loss3D()
        criterionExtra = losses.VGG16Loss3D()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True,)
            for idx, sample in enumerate(data_loader):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                # x => [batch, ch, frame, row, col]
                gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_ab], dim=1)
                # loss_l = self.criterion(gen_y[:,:1,:,:], y[:,:1,:,:])
                # loss_ab = self.criterion(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss_l = criterionA(gen_y[:,:1,:,:,:], y[:,:1,:,:,:])
                loss_ab = criterionB(gen_y[:,1:,:,:,:], y[:,1:,:,:,:])
                loss_lab = criterionExtra(gen_y, y)
                loss = loss_l + loss_ab + args.alpha * loss_lab
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                acc = []
                for f in range(y.shape[2]):
                    acc.append(self.evaluator(gen_y[:,:,f,:,:].detach(), y[:,:,f,:,:].detach()).item())
                trn_acc.append(sum(acc)/len(acc))
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3), 
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]
 
                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False,  float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True,)
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_ab], dim=1)
                loss, acc = [], []
                for f in range(y.shape[2]):
                    loss.append(self.criterion(gen_y[:,:,f,:,:].detach(), y[:,:,f,:,:].detach()).item())
                    acc.append(self.evaluator(gen_y[:,:,f,:,:].detach(), y[:,:,f,:,:].detach()).item())
                val_loss.append(sum(loss)/len(loss))
                val_acc.append(sum(acc)/len(acc))

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3), 
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class bergTrainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True,)
            for idx, sample in enumerate(data_loader):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_y = net(x)
                # loss = criterionB(gen_y, y)
                loss_l = criterionA(gen_y[:,:1,:,:], y[:,:1,:,:])
                loss_ab = criterionB(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss = loss_l + loss_ab
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3), 
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]
 
                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False,  float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True,)
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_y = net(x)
                loss_l = criterionA(gen_y[:,:1,:,:], y[:,:1,:,:])
                loss_ab = criterionB(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss = loss_l + loss_ab
                val_loss.append(loss.detach().item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3), 
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)
