import torch
import torch.nn as nn
import torch.nn.functional as F

from mvt.utils.init_util import normal_init
from ..block_builder import HEADS
from .base_cls_head import BaseClsDenseHead


@HEADS.register_module()
class LinearClsHeadClarity(BaseClsDenseHead):
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
                 stage_weight=-1,
                 out_stage=-1,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(LinearClsHeadClarity, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.sigmoid = torch.nn.Sigmoid()
        self.stage_weight = stage_weight
        self.out_stage = out_stage
        # self.mse_loss = torch.nn.MSELoss()
        self._init_layers()

    def _init_layers(self):
        if self.stage_weight == -1:
            self.fc = nn.Linear(self.in_channels, self.num_classes)
        else:
            self.fcs = [
                nn.Linear(256, self.num_classes),
                nn.Linear(512, self.num_classes),
                nn.Linear(1024, self.num_classes),
                nn.Linear(2048, self.num_classes)]

    def init_weights(self):
        if self.stage_weight == -1:
            normal_init(self.fc, mean=0, std=0.01, bias=0)
        else:
            for i in range(4):
                normal_init(self.fcs[i], mean=0, std=0.01, bias=0)

    def forward(self, x):
        if self.training:
            if self.stage_weight == -1:
                out = self.fc(x[self.out_stage])
            else:
                outs = []
                for i in range(4):
                    outs.append(self.fcs[i](x[i]))
                self.num_samples = len(x)
            return out
        else:
            return self.fc(feats[self.out_stage])
    
    def loss(self, x, gt_label):
        
        losses = dict()
        
        if isinstance(gt_label, list):
            gt_label = torch.tensor(gt_label, dtype=torch.long, device=gt_label[0].device)
        # compute loss and accuracy
        if self.stage_weight == -1:
            loss = self.compute_loss(x, gt_label, avg_factor=num_samples)
            acc = self.compute_accuracy(x, gt_label)
        else:
            stage_losses = []
            for i in range(4):
                stage_losses.append(self.stage_weight[i] * self.compute_loss(x[i], gt_label, avg_factor=num_samples))
            loss = sum(stage_losses)
            acc = self.compute_accuracy(x[self.out_stage], gt_label)

        assert len(acc) == len(self.topk)
        losses['loss'] = loss
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        if self.stage_weight != -1:
            for i in range(4):
                losses['accuracy']['stage_loss_{}'.format(i)] = stage_losses[i]

        return losses
