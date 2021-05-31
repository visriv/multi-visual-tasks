from .darknet import Darknet, CSPDarknet53
from .hourglass import HourglassModule, HourglassNet
from .hrnet import HRModule, HRNet
from .regnet import RegNet
from .resnet import (ResBlock, ResLayer, BasicBlock, 
                     Bottleneck, ResNet, ResNetV1c,
                     ResNetV1d)
from .resnext import ResNeXt
from .sdd_vgg import L2Norm, SSDVGG
from .vgg import VGG
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .googlenet import GoogLeNet
from .googlenet_clarity import GoogLeNetClarity
from .inception_v3 import InceptionV3
from .seresnet import SEResNet
from .tiny_yolo_v4 import TinyYOLOV4Net
from .yolo_v5 import YOLOV5Net
from .efficient import EfficientNet
from .inception_reg_net import InceptionRegNet
# from .swin_transformer import SwinTransformer
from .bit_models import ResNetV2


__all__ = [
    'Darknet', 'HourglassModule', 'HourglassNet', 'HRModule', 'HRNet',
    'RegNet', 'ResBlock', 'ResLayer', 'BasicBlock', 'Bottleneck', 
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'L2Norm', 'SSDVGG', 
    'VGG', 'MobileNetV2', 'InceptionV3', 
    'GoogLeNet', 'GoogLeNetClarity', 'SEResNet', 'MobileNetV3',
    'CSPDarknet53', 'TinyYOLOV4Net', 'YOLOV5Net', 'EfficientNet',
    # 'SwinTransformer', 
    'ResNetV2', 'InceptionRegNet'
]
