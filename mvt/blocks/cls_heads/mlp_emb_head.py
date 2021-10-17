import torch
import torch.nn as nn

from mvt.utils.init_util import normal_init
from mvt.cores.metric_ops.multi_similarity_miner import MultiSimilarityMiner
from ..block_builder import HEADS, build_loss
from .base_cls_head import BaseClsHead


@HEADS.register_module()
class MlpEmbHead(BaseClsHead):
    """Linear regressor head.

    Args:
        num_dim (int): Number of regression dimensions.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    def __init__(self, out_channels, in_channels, loss=dict(type="TripletMarginLoss")):
        super(MlpEmbHead, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = in_channels // 2
        self.out_channels = out_channels
        self.loss = build_loss(loss)
        self.miner = MultiSimilarityMiner()

        self._init_layers()

    def _init_layers(self):
        self.fc1 = nn.Linear(self.in_channels, self.mid_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.mid_channels, self.out_channels)

    def init_weights(self):
        normal_init(self.fc1, mean=0, std=0.01, bias=0)
        normal_init(self.fc2, mean=0, std=0.01, bias=0)

    def forward(self, x):
        x = x[0].view(x[0].size(0), -1)
        x = self.fc2(self.act(self.fc1(x)))
        return x

    def forward_train(self, feats, labels, output_emb=False):
        embeddings = self(feats)
        if isinstance(labels, list):
            cat_labels = torch.cat(labels)
        hard_pairs = self.miner(embeddings, cat_labels)
        losses = self.loss(embeddings, cat_labels, hard_pairs)
        if output_emb:
            return dict(loss_emb=losses), embeddings
        else:
            return dict(loss_emb=losses)
