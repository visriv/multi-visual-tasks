import torch
import torch.nn as nn
import torch.nn.functional as F

from mvt.utils.init_util import normal_init
from ..block_builder import HEADS
from .base_cls_head import BaseClsDenseHead


@HEADS.register_module()
class LinearClsHead(BaseClsDenseHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(LinearClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)
    
    def forward(self, x):
        x = self.fc(x)
        return x
