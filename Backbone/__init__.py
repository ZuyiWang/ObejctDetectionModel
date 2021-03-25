from .resnet import resnet101
from .vgg import vgg16

__all__ = ['build_backbone', 'resnet101']

model_zoo = {
    'resnet101': resnet101,
    'vgg16': vgg16,
}


def build_backbone(cfg):
    if cfg.MODEL.BACKBONE.NAME in model_zoo:
        return model_zoo[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
    else:
        raise Exception('Unknown model: ' + (cfg.MODEL.BACKBONE.NAME))
