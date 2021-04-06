# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from layers import *
# from data import voc, coco, voc_resnet
# import os
# from resnet.resnet import Bottleneck, ResNet
# from layers.modules.prediction_module import PredictionModule_C


# class SSD(nn.Module):
#     """Single Shot Multibox Architecture
#     The network is composed of a base VGG network followed by the
#     added multibox conv layers.  Each multibox layer branches into
#         1) conv2d for class conf scores
#         2) conv2d for localization predictions
#         3) associated priorbox layer to produce default bounding
#            boxes specific to the layer's feature map size.
#     See: https://arxiv.org/pdf/1512.02325.pdf for more details.

#     Args:
#         phase: (string) Can be "test" or "train"
#         basenet: 'vgg16' or 'resnet101'
#         size: input image size
#         base: VGG16 layers or ResNet model for input, size of either 300 or 500 for vgg16, 320 for resnet
#         extras: extra layers that feed to multibox loc and conf layers
#         head: "multibox head" consists of loc and conf conv layers
#         predict: new prediction modules
#     """

#     def __init__(self, phase, basenet, size, base, head, num_classes, extras=None):  # predict=None, 
#         super(SSD, self).__init__()
#         self.phase = phase
#         self.basenet = basenet
#         self.num_classes = num_classes
#         if basenet == 'vgg16':
#             self.cfg = (coco, voc)[num_classes == 21]
#         elif basenet == 'resnet101':
#             self.cfg = voc_resnet
#         self.priorbox = PriorBox(self.cfg)
#         # self.priors = Variable(self.priorbox.forward(), volatile=True)
#         self.priors = self.priorbox.forward()
#         self.priors.requires_grad = False
#         self.size = size

#         # SSD network
#         if basenet == 'vgg16':
#             self.vgg = nn.ModuleList(base)
#             # Layer learns to scale the l2 normalized features from conv4_3
#             self.L2Norm = L2Norm(512, 20)
#             self.extras = nn.ModuleList(extras)
#         elif basenet == 'resnet101':
#             self.resnet101 = base
#         # SSD prediction layers
#         self.loc = nn.ModuleList(head[0])
#         self.conf = nn.ModuleList(head[1])
#         # if head:
#         #     self.loc = nn.ModuleList(head[0])
#         #     self.conf = nn.ModuleList(head[1])
#         #     self.predict_modules = None
#         # elif predict:
#         #     self.predict_modules = predict
#         # else:
#         #     raise Exception('No head layers and prediction modules at the same time!')

#         if phase == 'test':
#             self.softmax = nn.Softmax(dim=-1)
#             self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

#     def forward(self, x):
#         """Applies network layers and ops on input image(s) x.

#         Args:
#             x: input image or batch of images. Shape: [batch,3,300,300].

#         Return:
#             Depending on phase:
#             test:
#                 Variable(tensor) of output class label predictions,
#                 confidence score, and corresponding location predictions for
#                 each object detected. Shape: [batch,topk,7]

#             train:
#                 list of concat outputs from:
#                     1: confidence layers, Shape: [batch*num_priors,num_classes]
#                     2: localization layers, Shape: [batch,num_priors*4]
#                     3: priorbox layers, Shape: [2,num_priors*4]
#         """
#         sources = list()
#         loc = list()
#         conf = list()
#         if self.basenet == 'vgg16':
#             # apply vgg up to conv4_3 relu
#             for k in range(23):
#                 x = self.vgg[k](x)

#             s = self.L2Norm(x)
#             sources.append(s)

#             # apply vgg up to fc7
#             for k in range(23, len(self.vgg)):
#                 x = self.vgg[k](x)
#             sources.append(x)

#             # apply extra layers and cache source layer outputs
#             for k, v in enumerate(self.extras):
#                 x = F.relu(v(x), inplace=True)
#                 if k % 2 == 1:
#                     sources.append(x)
#         elif self.basenet == 'resnet101':
#             p3, p5, p6, p7, p8, p9 = self.resnet101(x)
#             sources.extend([p3, p5, p6, p7, p8, p9])

#         # apply multibox head to source layers
#         for (x, l, c) in zip(sources, self.loc, self.conf):
#             loc.append(l(x).permute(0, 2, 3, 1).contiguous())
#             conf.append(c(x).permute(0, 2, 3, 1).contiguous())
#         # if not self.predict_modules:
#         #     for (x, l, c) in zip(sources, self.loc, self.conf):
#         #         loc.append(l(x).permute(0, 2, 3, 1).contiguous())
#         #         conf.append(c(x).permute(0, 2, 3, 1).contiguous())
#         # else:
#         #     for (x, pre_) in zip(sources, self.predict_modules):
#         #         c, l = pre_(x)
#         #         loc.append(l.permute(0, 2, 3, 1).contiguous())
#         #         conf.append(c.permute(0, 2, 3, 1).contiguous())                


#         loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
#         conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
#         if self.phase == "test":           
#             # output = self.detect(
#             #     loc.view(loc.size(0), -1, 4),                   # loc preds
#             #     self.softmax(conf.view(conf.size(0), -1,
#             #                 self.num_classes)),                # conf preds
#             #     self.priors.type(type(x.data))                  # default boxes
#             # )
#             # Detect类不支持静态方法的前向传播，因此使用类方法调用
#             output = self.detect.forward(
#                 loc.view(loc.size(0), -1, 4),                   # loc preds
#                 self.softmax(conf.view(conf.size(0), -1,
#                             self.num_classes)),                # conf preds
#                 self.priors.type(type(x.data))                  # default boxes
#             )
#         else:
#             output = (
#                 loc.view(loc.size(0), -1, 4),
#                 conf.view(conf.size(0), -1, self.num_classes),
#                 self.priors
#             )
#         return output

#     def load_weights(self, base_file):
#         other, ext = os.path.splitext(base_file)
#         if ext == '.pkl' or '.pth':
#             print('Loading weights into state dict...')
#             self.load_state_dict(torch.load(base_file,
#                                  map_location=lambda storage, loc: storage))
#             print('Finished!')
#         else:
#             print('Sorry only .pth and .pkl files supported.')


# # This function is derived from torchvision VGG make_layers()
# # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# def vgg(cfg, i, batch_norm=False):
#     """
#     Args:
#         cfg: vgg的配置文件
#         i: 输入图像的通道数
#         batch_norm: batch正则化
#     Return:
#         layers: vgg的包含各层的list
#     """
#     layers = []
#     in_channels = i
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         elif v == 'C':
#             # ceil_mode – when True, will use ceil instead of floor to compute the output shape
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#     conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
#     conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
#     layers += [pool5, conv6,
#                nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
#     return layers


# def add_extras(cfg, i, batch_norm=False):
#     # Extra layers added to VGG for feature scaling
#     layers = []
#     in_channels = i
#     flag = False
#     for k, v in enumerate(cfg):
#         if in_channels != 'S':
#             if v == 'S':
#                 layers += [nn.Conv2d(in_channels, cfg[k + 1],
#                            kernel_size=(1, 3)[flag], stride=2, padding=1)]
#             else:
#                 layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
#             flag = not flag
#         in_channels = v
#     return layers


# def multibox(vgg, extra_layers, cfg, num_classes):
#     loc_layers = []
#     conf_layers = []
#     # vgg的conv4_3和conv7的feature map用来做预测
#     vgg_source = [21, -2]
#     for k, v in enumerate(vgg_source):
#         loc_layers += [nn.Conv2d(vgg[v].out_channels,
#                                  cfg[k] * 4, kernel_size=3, padding=1)]
#         conf_layers += [nn.Conv2d(vgg[v].out_channels,
#                         cfg[k] * num_classes, kernel_size=3, padding=1)]
#     # 每个extra_layer有两层conv
#     for k, v in enumerate(extra_layers[1::2], 2):
#         loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
#                                  * 4, kernel_size=3, padding=1)]
#         conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
#                                   * num_classes, kernel_size=3, padding=1)]
#     return vgg, extra_layers, (loc_layers, conf_layers)

# def multihead_resnet(resnet, cfg, num_classes):
#     loc_layers = []
#     conf_layers = []
#     out_channels = (512, 1024, 1024, 1024, 1024, 1024)
#     # resnet使用conv3和conv5的最后一层的feature map用来做预测
#     for k in range(2):
#         loc_layers += [nn.Conv2d(out_channels[k],
#                                 cfg[k] * 4, kernel_size=3, padding=1)]
#         conf_layers += [nn.Conv2d(out_channels[k],
#                                 cfg[k] * num_classes, kernel_size=3, padding=1)]
#     # resnet包含的4个extra_layer
#     for k in range(2, 6):
#         loc_layers += [nn.Conv2d(out_channels[k], cfg[k]
#                                 * 4, kernel_size=3, padding=1)]
#         conf_layers += [nn.Conv2d(out_channels[k], cfg[k]
#                                 * num_classes, kernel_size=3, padding=1)]
#     return (loc_layers, conf_layers)

# def multipredict_resnet(resnet, cfg, num_classes):
#     pre_modules = []
#     # resnet101中选择的各feature map层的channel数
#     out_channels = (512, 1024, 1024, 1024, 1024, 1024)

#     # resnet使用conv3和conv5的最后一层的feature map用来做预测
#     for k in range(2):
#         pre_modules += [PredictionModule_C(out_channels[k], cfg[k], num_classes)]

#     # resnet包含的4个extra_layer
#     for k in range(2, 6):
#         pre_modules += [PredictionModule_C(out_channels[k], cfg[k], num_classes)]
#     return pre_modules

# vgg_base = {
#     '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
#             512, 512, 512],
#     '512': [],
# }
# extras = {
#     '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
#     '512': [],
# }
# mbox = {
#     '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
#     '512': [],
#     'resnet101': [4, 6, 6, 6, 4, 4],
# }
# resnet_base = {
#     'resnet101': [3, 4, 23, 3],
# }


# def build_ssd(phase, backbone, size=300, num_classes=21):
#     if phase != "test" and phase != "train":
#         print("ERROR: Phase: " + phase + " not recognized")
#         return
#     if size != 300 and size != 320:
#         print("ERROR: You specified size " + repr(size) + ". However, " +
#               "currently only SSD300 (size=300) or SSD with resnet101 (size=320) is supported!")
#         return
#     if backbone == 'vgg16':
#         base_, extras_, head_ = multibox(vgg(vgg_base[str(size)], 3),
#                                         add_extras(extras[str(size)], 1024),
#                                         mbox[str(size)], num_classes)
#         return SSD(phase, backbone, size, base_, head_, num_classes, extras_)
#     elif backbone == 'resnet101':
#         base_ = ResNet(Bottleneck, resnet_base[backbone])
#         # SSD中的预测卷积层
#         head_ = multihead_resnet(base_, mbox[backbone], num_classes)
#         # 修改后的预测模块
#         # prediction_ = multipredict_resnet(base_, mbox[backbone], num_classes)
#         # return SSD(phase, backbone, size, base_, None, num_classes, predict=prediction_)
#         return SSD(phase, backbone, size, base_, head_, num_classes)
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco, voc_resnet
import os
from resnet.resnet import Bottleneck, ResNet
from layers.modules.prediction_module import PredictionModule_C


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        basenet: 'vgg16' or 'resnet101'
        size: input image size
        base: VGG16 layers or ResNet model for input, size of either 300 or 500 for vgg16, 320 for resnet
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        predict: new prediction modules
    """

    def __init__(self, phase, basenet, size, base, head, num_classes, predict=None, extras=None):
        super(SSD, self).__init__()
        self.phase = phase
        self.basenet = basenet
        self.num_classes = num_classes
        if basenet == 'vgg16':
            self.cfg = (coco, voc)[num_classes == 21]
        elif basenet == 'resnet101':
            self.cfg = voc_resnet
        self.priorbox = PriorBox(self.cfg)
        # self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.priors = self.priorbox.forward()
        self.priors.requires_grad = False
        self.size = size

        # SSD network
        if basenet == 'vgg16':
            self.vgg = nn.ModuleList(base)
            # Layer learns to scale the l2 normalized features from conv4_3
            self.L2Norm = L2Norm(512, 20)
            self.extras = nn.ModuleList(extras)
        elif basenet == 'resnet101':
            self.resnet101 = base
        # SSD prediction layers
        if head:
            self.loc = nn.ModuleList(head[0])
            self.conf = nn.ModuleList(head[1])
            self.predict_modules = None
        elif predict:
            self.predict_modules = nn.ModuleList(predict)
        else:
            raise Exception('No head layers and prediction modules at the same time!')

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        if self.basenet == 'vgg16':
            # apply vgg up to conv4_3 relu
            for k in range(23):
                x = self.vgg[k](x)

            s = self.L2Norm(x)
            sources.append(s)

            # apply vgg up to fc7
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)
            sources.append(x)

            # apply extra layers and cache source layer outputs
            for k, v in enumerate(self.extras):
                x = F.relu(v(x), inplace=True)
                if k % 2 == 1:
                    sources.append(x)
        elif self.basenet == 'resnet101':
            p3, p5, p6, p7, p8, p9 = self.resnet101(x)
            sources.extend([p3, p5, p6, p7, p8, p9])

        # apply multibox head to source layers
        if not self.predict_modules:
            for (x, l, c) in zip(sources, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        else:
            for (x, pre_) in zip(sources, self.predict_modules):
                c, l = pre_(x)
                loc.append(l.permute(0, 2, 3, 1).contiguous())
                conf.append(c.permute(0, 2, 3, 1).contiguous())                


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":           
            # output = self.detect(
            #     loc.view(loc.size(0), -1, 4),                   # loc preds
            #     self.softmax(conf.view(conf.size(0), -1,
            #                 self.num_classes)),                # conf preds
            #     self.priors.type(type(x.data))                  # default boxes
            # )
            # Detect类不支持静态方法的前向传播，因此使用类方法调用
            output = self.detect.forward(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                            self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    """
    Args:
        cfg: vgg的配置文件
        i: 输入图像的通道数
        batch_norm: batch正则化
    Return:
        layers: vgg的包含各层的list
    """
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            # ceil_mode – when True, will use ceil instead of floor to compute the output shape
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    # vgg的conv4_3和conv7的feature map用来做预测
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    # 每个extra_layer有两层conv
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

def multihead_resnet(resnet, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    out_channels = (512, 1024, 1024, 1024, 1024, 1024)
    # resnet使用conv3和conv5的最后一层的feature map用来做预测
    for k in range(2):
        loc_layers += [nn.Conv2d(out_channels[k],
                                cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(out_channels[k],
                                cfg[k] * num_classes, kernel_size=3, padding=1)]
    # resnet包含的4个extra_layer
    for k in range(2, 6):
        loc_layers += [nn.Conv2d(out_channels[k], cfg[k]
                                * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(out_channels[k], cfg[k]
                                * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)

def multipredict_resnet(resnet, cfg, num_classes):
    pre_modules = []
    # resnet101中选择的各feature map层的channel数
    out_channels = (512, 1024, 1024, 1024, 1024, 1024)

    # resnet使用conv3和conv5的最后一层的feature map用来做预测
    for k in range(2):
        pre_modules += [PredictionModule_C(out_channels[k], cfg[k], num_classes)]

    # resnet包含的4个extra_layer
    for k in range(2, 6):
        pre_modules += [PredictionModule_C(out_channels[k], cfg[k], num_classes)]
    return pre_modules

vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
    'resnet101': [4, 6, 6, 6, 4, 4],
}
resnet_base = {
    'resnet101': [3, 4, 23, 3],
}


def build_ssd(phase, backbone, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300 and size != 320:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) or SSD with resnet101 (size=320) is supported!")
        return
    if backbone == 'vgg16':
        base_, extras_, head_ = multibox(vgg(vgg_base[str(size)], 3),
                                        add_extras(extras[str(size)], 1024),
                                        mbox[str(size)], num_classes)
        return SSD(phase, backbone, size, base_, head_, num_classes, extras_)
    elif backbone == 'resnet101':
        base_ = ResNet(Bottleneck, resnet_base[backbone])
        # SSD中的预测卷积层
        # head_ = multihead_resnet(base_, mbox[backbone], num_classes)
        # 修改后的预测模块
        prediction_ = multipredict_resnet(base_, mbox[backbone], num_classes)
        return SSD(phase, backbone, size, base_, None, num_classes, predict=prediction_)
        # return SSD(phase, backbone, size, base_, head_, num_classes)
        

    

    
