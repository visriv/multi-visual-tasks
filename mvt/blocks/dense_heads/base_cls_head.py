from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn

from ..block_builder import build_loss
from mvt.utils.metric_util import Accuracy


class BaseClsDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(BaseClsDenseHead, self).__init__()
        if isinstance(topk, list):
            topk = tuple(topk)

        if isinstance(topk, list):
            topk = tuple(topk)
        print(topk)

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))


        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        
        if isinstance(gt_label, list):
            gt_label = torch.tensor(gt_label, dtype=torch.long, device=gt_label[0].device)
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        # compute accuracy
        acc = self.compute_accuracy(cls_score, gt_label)
        assert len(acc) == len(self.topk)
        losses['loss'] = loss
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        return losses

    def simple_test(self, x):
        """Test without augmentation."""
        cls_score = self(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
    
    def forward_train(self, x, gt_label, **kwargs):
        out = self(x)
        losses = self.loss(out, gt_label, **kwargs)
        return losses
