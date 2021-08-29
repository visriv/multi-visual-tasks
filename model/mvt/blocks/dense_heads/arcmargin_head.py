import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from .base_cls_head import BaseClsHead
from ..block_builder import HEADS, build_loss


@HEADS.register_module()
class ArcMarginHead(BaseClsHead):
    """Aditive angular margin for arcface/insightface
       Reference: Deng, Jiankang, et al. “ArcFace: Additive Angular Margin Loss for Deep Face Recognition.” CVPR, 2019.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of classification dimensions.
        scale (float): norm of input feature
        margin (float): coefficient used in cos(theta + margin)
        loss (dict): Config of classification loss.
    """

    def __init__(
        self,
        in_channels=128,
        out_channels=10575,
        scale=80.0,
        margin=0.50,
        easy_margin=False,
        loss=dict(type="CrossEntropyLoss"),
    ):
        super(ArcMarginHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(out_channels, in_channels))

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.loss = build_loss(loss)

    def init_weights(self):
        nn.init.xavier_uniform_(self.weight)

    def forward_train(self, x, labels):
        cosine = self(x)

        if isinstance(labels, list):
            cat_labels = torch.cat(labels)

        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, cat_labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        losses = self.loss(output, cat_labels)
        return dict(loss_cls=losses)

    def forward(self, x):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        return cosine
