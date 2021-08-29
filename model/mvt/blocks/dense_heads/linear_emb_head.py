import torch
import torch.nn as nn

from .base_emb_head import BaseEmbHead
from ..block_builder import HEADS, build_loss
from model.mvt.utils.init_util import normal_init
from model.mvt.cores.metric_ops.multi_similarity_miner import MultiSimilarityMiner


@HEADS.register_module()
class LinearEmbHead(BaseEmbHead):
    """Linear regressor head.

    Args:
        num_dim (int): Number of regression dimensions.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    def __init__(self,
                 out_channels,
                 in_channels,
                 loss=dict(type='TripletMarginLoss')):
        super(LinearEmbHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = build_loss(loss)
        self.miner = MultiSimilarityMiner()


        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.out_channels)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)
    
    def forward(self, x):
        x = x[0].view(x[0].size(0), -1)
        return self.fc(x)

    def forward_train(self, feats, labels):
        embeddings = self(feats)
        if isinstance(labels, list):
            cat_labels = torch.cat(labels)
        hard_pairs = self.miner(embeddings, cat_labels)
        losses = self.loss(embeddings, cat_labels, hard_pairs)
        return dict(loss_emb=losses)
