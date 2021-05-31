import torch
import numpy as np
from torch import nn
from collections import namedtuple
import torch.nn.functional as F
from collections import OrderedDict

from mvt.utils.metric_util import Accuracy
from mvt.cores.layer_ops.inception_layer import InceptionGAux

from ..block_builder import HEADS, build_loss
from .base_cls_head import BaseClsDenseHead


_GoogLeNetOuputs = namedtuple('GoogLeNetOuputs', ['logits', 'aux_logits2', 'aux_logits1'])


@HEADS.register_module()
class GoogLeNetClsHead(BaseClsDenseHead):
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
        super(GoogLeNetClsHead, self).__init__(loss, topk)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.aux_logits = aux_logits

        if aux_logits:
            self.aux1 = InceptionGAux(512, num_classes)
            self.aux2 = InceptionGAux(528, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)
    
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

        if self.training and self.aux_logits:
            aux1 = self.aux1(x[0])
            aux2 = self.aux2(x[1])
        else:
            out = x[2]
        
        x = self.avgpool(x[2])
        # N x 1024 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x num_classes

        if self.training and self.aux_logits:
            return _GoogLeNetOuputs(x, aux2, aux1)
        return x
    
    def loss(self, out, gt_label, **kwargs):
        if isinstance(out, tuple) and len(out) == 3:
            out, aux_out_2, aux_out_1 = out
        else:
            aux_out_1 = aux_out_2 = None
           
        num_samples = len(out)
        losses = dict()
        
        if isinstance(gt_label, list):
            gt_label = torch.tensor(gt_label, dtype=torch.long, device=gt_label[0].device)

        cls_loss = self.ce_loss(out, gt_label)
 
        if (aux_out_1 is not None) and (aux_out_2 is not None):
            aux_1_loss = 0.1*self.ce_loss(aux_out_1, gt_label)
            aux_2_loss = 0.1*self.ce_loss(aux_out_2, gt_label)
            loss = (cls_loss + aux_1_loss + aux_2_loss) / float(num_samples)
        else:
            loss = cls_loss / float(num_samples)

        acc = self.compute_accuracy(out, gt_label)

        losses['loss'] = loss
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}

        return losses
