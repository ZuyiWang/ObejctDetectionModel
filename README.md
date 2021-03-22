# ObjectDetectionModel
一些目标检测领域经典检测器的轮子，主要帮助自己理解一些重要的目标检测特征提取算法的原理与实现，基于python3.6+pytorch1.7.1， 使用Tensorboard进行训练过程的可视化

## Histograms of Oriented Gradients
+ 数据集下载：
   1. 本代码运行跑的数据集，但不是论文中训练测试的官方数据集：[ The CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)
   2. 论文中训练测试的数据集: [INRIA](http://lear.inrialpes.fr/data)
+ 论文地址：[Histograms of oriented gradients for human detection](https://ieeexplore.ieee.org/abstract/document/1467360)
## Deformable Part Model
+ 论文地址：
   1. [A DisCriminatively Trained. Multiscale, Deformable Part Model (CVPR 2008)](https://ieeexplore.ieee.org/document/4587597)
   2. [Object Detection with Discriminatively Trained Part-Based Models (PAMI 2010)](https://ieeexplore.ieee.org/document/5255236)
+ voc-release5: 作者给出的官方code， matlab和C++；[下载地址](http://www.rossgirshick.info/latent/)<br>
                Operation Environment: Matlab_R2020b+Image Processing ToolBox+gcc 7.5.0; 由于计算机64位和32位的区别，代码需要修改一下；
## YOLO_v1
+ 论文地址：
   1. [You only look once: Unified, real-time object detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)
+ 代码：
   1. 大部分代码fork自 https://github.com/abeardear/pytorch-YOLO-v1, 针对其中的一些bug以及自己的环境进行了修改与调整
## SSD
+ 论文地址：
   1. [SSD: Single Shot MultiBox Detector](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2)
+ 代码：
   1. 大部分代码fork自 https://github.com/amdegroot/ssd.pytorch, 针对其中的一些bug以及自己的环境进行了修改与调整
   2. 添加了对单张图片的检测, predict.py文件

# TODO
## DSSD

