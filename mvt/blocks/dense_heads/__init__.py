from .anchor_free import AnchorFreeHead
from .anchor import AnchorHead
from .atss import ATSSHead
from .fcos import FCOSHead
from .retina import RetinaHead
from .rpn import RPNHead
from .ssd import SSDHead
from .yolo import YOLOV3Head, TinyYOLOV4Head, YOLOV5Head
from .googlenet_cls_head_clarity import GoogLeNetClsHeadClarity
from .googlenet_cls_head import GoogLeNetClsHead
from .inception_cls_head_multitask import InceptionClsHeadMultitask
from .inception_cls_head import InceptionClsHead
from .linear_cls_head_clarity import LinearClsHeadClarity
from .linear_cls_head import LinearClsHead
from .linear_cls_head_clarity import LinearClsHeadClarity
from .retina_sepconv_head import RetinaSepConvHead
from .linear_reg_head import LinearRegHead


__all__ = [
    'AnchorFreeHead', 'AnchorHead', 
    'RPNHead', 'RetinaHead', 'SSDHead', 'FCOSHead', 
    'ATSSHead', 'YOLOV3Head', 'TinyYOLOV4Head', 'YOLOV5Head', 
    'GoogLeNetClsHeadClarity', 'GoogLeNetClsHead', 'InceptionClsHeadMultitask', 
    'InceptionClsHead', 'LinearClsHead', 'LinearClsHeadClarity',
    'RetinaSepConvHead', 'LinearRegHead'
]
