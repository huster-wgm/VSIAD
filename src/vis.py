#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import cv2
import PIL
import glob
import random
import torchvision
import numpy as np
import utils
from skimage.io import imread, imsave
import argparse


DIR = os.path.dirname(os.path.abspath(__file__))
Result_DIR = os.path.join(DIR, '../result/')
if not os.path.exists(os.path.join(Result_DIR, 'video')):
    os.mkdir(os.path.join(Result_DIR, 'video'))


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


SCENE_CATEGORY = {
    "VNIR": ["Scene{}".format(i) for i in range(6,11)],
    "NIR": ["Scene{}".format(i) for i in range(1,6)],
}


whitespace = 4


def add_whitespace(img):
    """
    Args:
        img : ndarray [h,w,c]
    """
    row, col, ch = img.shape
    tmp = np.ones((row + 2 * whitespace, col + 2 * whitespace, ch), "uint8") * 255
    tmp[whitespace:row + whitespace,
        whitespace:col + whitespace, :] = img
    return tmp


def add_label(img, label):
    """
    Args:
        img : ndarray [h,w,c]
        label : str
    """
    size = 2 # text size in pixels
    thickness = 2
    row, col, ch = img.shape
    dataset = ['VNIR', 'NIR', 'VC']
    pos = (whitespace+5, whitespace+50) # starting pos
    # if label in dataset:
    #     pos = (col//2 + whitespace+5, whitespace+50) # starting pos
    color = (255, 0, 0) # RGB => black
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, label, pos, font, size, color, thickness)
    return img


def get_notion(folder, sce, mode='file'):
    strs = folder.lower().split('-')
    if len(strs) == 2:
        # data, e.g., VC24-src, pix2pixHD
        network, ver = strs
        if 'pix2pix' in network:
            notion = 'pix2pixHD'
        else:
            if ver == "src" and sce in SCENE_CATEGORY['VNIR']:
                notion = "VNIR"
            elif ver == "src" and sce in SCENE_CATEGORY['NIR']:
                notion = "NIR"
            elif ver == "tar":
                notion = "VC"
    elif len(strs) == 3:
        # model, e.g., res18ynetsync-VC24-RGB2LAB
        network, root, ver = strs
        if 'berg' in network:
            notion = 'Berg et al.' if mode != 'file' else 'Berg'
        elif 'zhang' in network:
            notion = 'Zhang et al.' if mode != 'file' else 'Zhang'
        elif 'lizuka' in network:
            notion = 'Iizuka et al.' if mode != 'file' else 'Iizuka'
        # elif '3d' in network:
        #     notion = 'Ours3D'
        # elif 'res18ynetsync' in network:
        #     notion = 'Ours'
        #     if "smoothsmooth" in ver:
        #         notion = 'Ours(SmoothX2)'
        #     elif "smooth" in ver:
        #         notion = 'Ours(SmoothX1)'
        else:
            notion = "{}-{}".format(network, ver.upper())
    else:
        raise ValueError("Patern of {} don't exist.".format(folder))
    return notion


def img2video(imgsets, path, fps):
    """
    :type list[list[2d array]]
    :rtype mp4
    """
    
    height, width, ch = imgsets[0][0].shape
    use_empty = False
    empty = np.ones((height, width, ch), np.uint8) * 0
    if len(imgsets) > 3:
        if len(imgsets) % 2 == 0:
            size = (width * len(imgsets) // 2 + 2 * whitespace, 2 * height + 2 * whitespace)
        else:
            use_empty = True
            size = (width * (len(imgsets) // 2 + 1) + 2 * whitespace, 2 * height + 2 * whitespace)
    else:
        size = (width * len(imgsets) + 2 * whitespace, height + 2 * whitespace)

    out = cv2.VideoWriter(os.path.join(Result_DIR, 'video', '{}.mp4'.format(path)),
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(imgsets[0])):
        imgset = []
        for j in range(len(imgsets)):
            imgset.append(imgsets[j][i])
        if use_empty:
            imgset.append(empty)
        if len(imgset) % 2 == 0:
            img = np.concatenate(
                [np.concatenate(imgset[0::2], axis=1),
                 np.concatenate(imgset[1::2], axis=1)], axis=0)
        else:
            img = np.concatenate(imgset, axis=1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = add_whitespace(img)
        out.write(img)
    out.release()


def load_model(cuda):
    """
    :type boolean
    :rtype pytorch model
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    if cuda:
        model.cuda()
    return model


def get_prediction(path, model, cuda, threshold):
    """
    :type str
    :type pytorch model
    :type float
    :rtype numpy array
    """
    img = PIL.Image.open(path).convert('RGB')
    img = torchvision.transforms.functional.to_tensor(img)
    if cuda:
        img = img.cuda()
    pred = model([img])[0]
    for subset in ['scores','masks','labels','boxes']:
        if cuda:
            pred[subset] = pred[subset].detach().cpu()
        else:
            pred[subset] = pred[subset].detach()

    pred_score = list(pred['scores'].numpy())
    pred_t = -1
    for idx, score in enumerate(pred_score):
        if score > threshold:
            pred_t = idx
    masks = (pred['masks']>0.5).numpy()[:,0,:,:]
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class


def set_colour_masks(img, idx):
    """
    :type 2d array
    :rtype 3d array
    """
    colours = [[0, 255, 0],[0, 0, 255],
               [255, 0, 0],[0, 255, 255],
               [255, 255, 0],[255, 0, 255],
               [80, 70, 180],[250, 80, 190],
               [245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(img).astype(np.uint8)
    g = np.zeros_like(img).astype(np.uint8)
    b = np.zeros_like(img).astype(np.uint8)
    r[img == 1], g[img == 1], b[img == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask, colours[random.randrange(0,10)]


def detection(path, model, cuda, threshold=0.5, rect_th=2, text_size=0.5, text_th=2):
    """
    :type str
    :type dict tensor
    :type float
    :rtype pytorch model
    """
    masks, boxes, pred_cls = get_prediction(path, model, cuda, threshold)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask, color = set_colour_masks(masks[i], i)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=color, thickness=rect_th)
        text_pos = (boxes[i][0][0], boxes[i][1][1])
        cv2.putText(img, pred_cls[i], text_pos, cv2.FONT_HERSHEY_SIMPLEX, text_size, color, thickness=text_th)
    return img


def main(args):
    # load model
    if args.detect:
        model = load_model(args.cuda)
    # args.folders = ['VC24-src', 'VC24-tar'] + args.folders
    positions = ["Position{}".format(i) for i in range(0, 4)]
    scenes = ['Scene5', 'Scene10']
    for pos in positions:
        for sce in scenes:
            imgColorSets, imgDetectSets = [], []
            notions = []
            for folder in args.folders:
                print("Handling {}/{} ...".format(folder, sce))
                notions.append(get_notion(folder, sce))
                imgColors, imgDetects, imgLabel = [], [], []
                files = utils.natural_sort(
                    glob.glob(os.path.join(Result_DIR, 'frame', folder, '{}_{}_*.png'.format(pos, sce))))
                if len(files) > 0:
                    for idx, file in enumerate(files):
                        label = get_notion(folder, sce, 'video')
                        print("\t {}/{} >> {}".format(
                            folder, os.path.basename(file), label), end='\r', flush=True)
                        img = np.array(PIL.Image.open(file).convert('RGB'))
                        if args.label:
                            imgColors.append(
                                add_label(add_whitespace(img), label))
                            if args.detect:
                                img = detection(file, model, args.cuda)
                                imgDetects.append(
                                    add_label(add_whitespace(img), label))
                        else:
                            imgColors.append(add_whitespace(img))
                            if args.detect:
                                img = detection(file, model, args.cuda)
                                imgDetects.append(add_whitespace(img))
                    imgColorSets.append(imgColors)
                    if args.detect:
                        imgDetectSets.append(imgDetects)
                else:
                    "Provided folder: {} is empty.".format(folder)

            notion = '&'.join(notions)
            if len(files) > 0:
                img2video(imgColorSets, '{}-color-{}-{}'.format(notion, pos, sce), args.fps)
                if args.detect:
                    img2video(imgDetectSets, '{}-detect-{}-{}'.format(notion, pos, sce), args.fps)


if __name__ == "__main__":
    # setup parameters
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-folders', nargs='+', type=str, default=["res18ynet-Seq3B-RGB2LAB"],
                        help='results used for detections ')
    parser.add_argument('-root', type=str, default='Seq3B',
                        help='root dir of original files ')
    parser.add_argument('-fps', type=int, default=24,
                        help='FPS of generated videos ')
    parser.add_argument('-label', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='video with label on top')
    parser.add_argument('-detect', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='using detection')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()
    main(args)