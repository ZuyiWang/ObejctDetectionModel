from . import resnet

# 将需要的模型放入__all__, 定义相应的函数
__all__ = ['resnet101']


def resnet101(cfg, pretrained=True):
    return resnet.resnet101(pretrained=pretrained)
