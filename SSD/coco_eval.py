from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import COCO_ROOT, COCOAnnotationTransform, COCODetection, BaseTransform
from data import COCO_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/COCO_vgg16.pth', type=str,                # VOC_resnet101_CNN+pre_95000
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='coco_eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--coco_root', default="/media/iaes/新加卷/wangzy/COCO2017",
                    help='Location of COCO root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.coco_root, 'annotations', 'instances_val2017.json')
imgpath = os.path.join(args.coco_root, 'val2017', '%s.jpg')
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    # all_boxes = [[[] for _ in range(num_images)]
    #              for _ in range(len(labelmap)+1)]
    all_boxes = []

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300', 'coco_'+set_type)
    det_file = os.path.join(output_dir, 'detections.json')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        img_id = dataset.ids[i]

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            category_id = list(dataset.target_transform.label_map.keys())[list(dataset.target_transform.label_map.values()).index(j)]
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]

            scores = dets[:, 0].cpu().numpy()
            # cls_dets = np.hstack((boxes.cpu().numpy(),
            #                       scores[:, np.newaxis])).astype(np.float32,
            #                                                      copy=False)
            # all_boxes[j][i] = cls_dets
            for box, score in zip(boxes, scores):
                all_boxes.append({'image_id':img_id, 'category_id':category_id, 'bbox':[round(i, 2) for i in box.cpu().numpy().tolist()], 'score':round(float(score), 3)})

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

    with open(det_file, 'w') as f:
        json.dump(all_boxes, f)

    print('Evaluating detections')
    # evaluate_detections(all_boxes, output_dir, dataset)
    cocoGt = COCO(annopath)        #标注文件的路径及文件名，json文件形式
    cocoDt = cocoGt.loadRes(det_file)       #自己的生成的结果的路径及文件名，json文件形式
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', 'vgg16', 300, num_classes)            # initialize SSD  resnet101
    net.load_state_dict(torch.load(args.trained_model, map_location='cuda:0'))   # 多GPU训练的结果，测试时需要map_location指定GPU
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = COCODetection(root=args.coco_root, image_set='val2017',
                            transform=BaseTransform(300, dataset_mean), 
                            target_transform=COCOAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)
