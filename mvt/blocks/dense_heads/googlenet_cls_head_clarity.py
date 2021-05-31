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
class GoogLeNetClsHeadClarity(BaseClsDenseHead):
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
        super(GoogLeNetClsHeadClarity, self).__init__(loss, topk)
        self.mse_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.aux_logits = aux_logits

        if aux_logits:
            self.aux1 = InceptionGAux(512, num_classes, sigmoid=False)
            self.aux2 = InceptionGAux(528, 1, sigmoid=True)

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
            out = _GoogLeNetOuputs(x[2], aux2, aux1)
        else:
            out = x[2]
        
        return out

    def loss(self, out, gt_label, scale_gt_label, **kwargs):
        if isinstance(out, tuple) and len(out) == 3:
            out, aux_out_2, aux_out_1 = out
        else:
            aux_out_1 = aux_out_2 = None

        scale_gt_label = scale_gt_label.float()
           
        num_samples = len(out)
        losses = dict()
        
        loss_scale = 0.1
        if isinstance(gt_label, list):
            gt_label = torch.tensor(gt_label, dtype=torch.long, device=gt_label[0].device)

        reg_loss = loss_scale * 33 * self.mse_loss(out, scale_gt_label)
 
        if (aux_out_1 is not None) and (aux_out_2 is not None):
            aux_cls_loss = loss_scale * 0.1 * self.ce_loss(aux_out_1, gt_label)
            aux_reg_loss = loss_scale * 10 * self.mse_loss(aux_out_2, scale_gt_label)
            loss = (reg_loss + aux_cls_loss + aux_reg_loss) / float(num_samples)
        else:
            loss = reg_loss / float(num_samples)

        pred = torch.clamp(torch.round(out.view(-1)*10-0.7), min=0, max=9)
        correct_1 = torch.eq(torch.clamp(torch.abs(pred - gt_label), min=0), 0.0)
        correct_2 = torch.eq(torch.clamp(torch.abs(pred - gt_label)-1, min=0), 0.0)
        acc_top_1 = torch.sum(correct_1)/float(num_samples)
        acc_top_2 = torch.sum(correct_2)/float(num_samples)
        
        losses['loss'] = loss
        msgs = {
            'reg_loss': reg_loss,
            'top-1': acc_top_1, 
            'top-2': acc_top_2
        }
        if aux_out_1 is not None:
            msgs['aux_cls_loss'] = aux_cls_loss
            msgs['aux_reg_loss'] = aux_reg_loss
        losses['accuracy'] = msgs
        return losses
