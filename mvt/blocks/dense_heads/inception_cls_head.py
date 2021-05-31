import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from mvt.utils.metric_util import Accuracy
from mvt.cores.layer_ops.inception_layer import InceptionMAux

from ..block_builder import HEADS, build_loss
from .base_cls_head import BaseClsDenseHead


@HEADS.register_module()
class InceptionClsHead(BaseClsDenseHead):
    """classification aux logits head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
    """

    def __init__(self,
                 num_classes=1000, 
                 aux_logits=True, 
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(InceptionClsHead, self).__init__(loss, topk)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.aux_logits = aux_logits

        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.group1 = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(2048, num_classes))
            ])
        )
    
    def init_weights(self):        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                m.weight.data.copy_(values.reshape(m.weight.shape))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.group1(x[1])
        
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x[0])
            return x, aux
        else:
            return x
    
    def loss(self, out, gt_label, **kwargs):
        if isinstance(out, tuple) and len(out) == 2:
            out, aux_out = out
        else:
            aux_out = None
           
        num_samples = len(out)
        losses = dict()
        
        if isinstance(gt_label, list):
            gt_label = torch.tensor(gt_label, dtype=torch.long, device=gt_label[0].device)

        cls_loss = self.ce_loss(out, gt_label)
 
        if aux_out is not None:
            aux_loss = 0.1*self.ce_loss(aux_out, gt_label)
            loss = (cls_loss + aux_loss) / float(num_samples)
        else:
            loss = cls_loss / float(num_samples)

        acc = self.compute_accuracy(out, gt_label)

        losses['loss'] = loss
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}

        return losses
