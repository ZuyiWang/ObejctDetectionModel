from . import vgg

# 将需要的模型放入__all__, 定义相应的函数
__all__ = ['vgg16']


def vgg16(cfg, pretrained=True):
    return vgg.vgg16(pretrained=pretrained)
