import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_reg_head import BaseRegHead
from ..block_builder import HEADS, build_loss
from mvt.utils.init_util import normal_init


@HEADS.register_module()
class LinearRegHead(BaseRegHead):
    """Linear regressor head.

    Args:
        num_dim (int): Number of regression dimensions.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    def __init__(self,
                 out_dim,
                 in_channels,
                 loss=dict(type='MSELoss')):
        super(LinearRegHead, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.loss = build_loss(loss)

        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.out_dim)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

    