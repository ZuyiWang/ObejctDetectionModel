"""
predict an image using SSD
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
# from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform, base_transform
from data import VOC_CLASSES as labelmap
from data import COCO_ROOT, COCOAnnotationTransform, COCODetection, BaseTransform, base_transform
# from data import COCO_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
# parser.add_argument('--trained_model',
#                     default='weights/ssd300_mAP_77.43_v2.pth', type=str,
#                     help='Trained state_dict file path to open')
parser.add_argument('--trained_model',
                    default='weights/VOC_vgg16.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--confidence_threshold', default=0.5, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--image_file', default="/home/iaes/ObjectDetectionModel/SSD/dog.jpg",
                    help='Location of image')

args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset_mean = (104, 117, 123)
set_type = 'test'
Color = [[0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]]




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

def predict_net(net, cuda, image, top_k, height, width, im_size=300, thresh=0.05):
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[]for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    # height, width, channels = image.shape
    image = torch.from_numpy(image).permute(2, 0, 1)
    x = Variable(image.unsqueeze(0))
    if args.cuda:
        x = x.cuda()
    _t['im_detect'].tic()
    detections = net(x).data
    detect_time = _t['im_detect'].toc(average=False)

    # skip j = 0, because it's the background class
    for j in range(1, detections.size(1)):
        dets = detections[0, j, :]
        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)
        if dets.size(0) == 0:
            continue
        boxes = dets[:, 1:]
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        scores = dets[:, 0].cpu().numpy()
        cls_dets = np.hstack((boxes.cpu().numpy(),
                            scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
        all_boxes[j] = cls_dets

    print('im_detect: {:.3f}s'.format(detect_time))
    original_img = cv2.imread(args.image_file)
    for index, cls_name in enumerate(labelmap):
        if all_boxes[index+1] == []:
            continue
        for i in range(all_boxes[index+1].shape[0]):
            if all_boxes[index+1][i][-1] >= thresh:
                color = Color[(index+1)%21]
                left_up = (round(all_boxes[index+1][i][0]), round(all_boxes[index+1][i][1]))
                right_bottom = (round(all_boxes[index+1][i][2]), round(all_boxes[index+1][i][3]))
                cv2.rectangle(original_img, left_up, right_bottom, color, 2)
                label = cls_name + str(round(all_boxes[index+1][i][-1], 2))
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                p1 = (round(all_boxes[index+1][i][0]), round(all_boxes[index+1][i][1])- text_size[1])
                cv2.rectangle(original_img, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
                cv2.putText(original_img, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)  

    while True:
        cv2.imshow('result', original_img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break          

if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', 'vgg16', 300, num_classes)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model, map_location='cuda:0'), strict=False)
    net.eval()
    print('Finished loading model!')
    # load data
    img = cv2.imread(args.image_file)
    height, width, channels = img.shape
    img = base_transform(img, 300, dataset_mean)

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    predict_net(net, args.cuda, img, args.top_k, height, width, 300,
             thresh=args.confidence_threshold)
