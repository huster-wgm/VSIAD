#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import argparse
import os
import glob
import torch
import time
import numpy as np
import pandas as pd
import metrics
import utils
from skimage.io import imread, imsave


DIR = os.path.dirname(os.path.abspath(__file__))
Result_DIR = os.path.join(DIR, '../result')
if not os.path.exists(os.path.join(Result_DIR, 'raw')):
    os.mkdir(os.path.join(Result_DIR, 'raw'))

nirScenes = ['Scene{}'.format(i) for i in range(1, 6)]
vnirScenes = ['Scene{}'.format(i) for i in range(6, 11)]

def main(args):
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")

    # set up evaluators
    evaluators = [metrics.AE(), metrics.MSE(), metrics.PSNR(), metrics.SSIM(),
                  metrics.LPIPS(args.cuda), metrics.FID(args.cuda)]
    pred_dir = os.path.join(Result_DIR, 'frame', args.dir)
    gt_dir = os.path.join(Result_DIR, 'frame', '{}-tar'.format(args.dir.split('-')[1]))
    pFiles = utils.natural_sort(glob.glob(os.path.join(pred_dir, '*.png')))
    gFiles = utils.natural_sort(glob.glob(os.path.join(gt_dir, '*.png')))
    valid = utils.pair_validate(pFiles, gFiles)
    print('Evaluating >> {}'.format(args.dir))
    if valid:
        performs = []
        for pfile, gfile in zip(pFiles, gFiles):
            filename = os.path.basename(pfile)
            pos, sce = filename.split('_')[:2]
            if sce in nirScenes:
                sce = 'NIR'
            elif sce in vnirScenes:
                sce = 'VNIR'
            else:
                raise ValueError("{} is not supported".format(sce))
            pimg = (imread(pfile) / 255).astype('float32')
            gimg = (imread(gfile) / 255).astype('float32')
            gen_y = torch.from_numpy(np.expand_dims(pimg.transpose(2,0,1), axis=0))
            y = torch.from_numpy(np.expand_dims(gimg.transpose(2,0,1), axis=0))
            if args.cuda:
                gen_y = gen_y.cuda()
                y = y.cuda()

            record = [pos, sce]
            for i, evaluator in enumerate(evaluators):
                record.append(evaluator(gen_y, y).item())

            performs += record
            print('\t Handling >> {} @ {} / {}'.format(os.path.basename(pfile), len(performs), len(pFiles)), 
                  end='\r', flush=True)

        headers = ['position', 'scene'] + [repr(x) for x in evaluators]
        performs = np.array(performs).reshape(-1, len(evaluators)+2)
        mp = list(np.mean(performs[:, 2:].astype("float32"),axis=0))
        assert performs.shape[0] == len(pFiles)
        performs = pd.DataFrame(performs,
                                columns=headers)
        performs.to_csv(os.path.join(Result_DIR, "raw", "{}.csv".format(args.dir)),
                        index=False, 
                        float_format="%.3f")
    
        # save mean performance
        log_path = os.path.join(Result_DIR, "Performs.csv")
        if os.path.exists(log_path):
            mperform = pd.read_csv(log_path)
        else:
            mperform = pd.DataFrame([])
        mpheaders = ['checkpoint'] + [repr(x) for x in evaluators]
        mp = pd.DataFrame([[args.dir] + mp],
                          columns=mpheaders)
        mperform = mperform.append(mp, ignore_index=True)
        mperform.to_csv(log_path, index=False, float_format="%.3f")


if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-dir', type=str, default=["res18unet-S6B-RGB2RGB"],
                        help='dir of prediction result ')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()

    main(args)
