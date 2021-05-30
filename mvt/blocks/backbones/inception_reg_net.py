import logging
import torch
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mtl.cores.ops import ConvModule
from mtl.utils.init_util import constant_init, kaiming_init
from mtl.utils.checkpoint_util import load_checkpoint
from ..block_builder import BACKBONES


@BACKBONES.register_module()
class InceptionRegNet(nn.Module):
    """InceptionRegNet backbone.

    Args:
        in_channels (list of int): input list of feature maps.
        branch_out_dim (int): Output dim for each branch
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
    """

    def __init__(self,
                 in_channels=[110, 74],
                 branch_out_dim=64,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU6')):
        super(InceptionRegNet, self).__init__()
        self.in_channels = in_channels
        self.branch_out_dim = branch_out_dim

        cfg_dict = {
            'conv_cfg': conv_cfg,
            'norm_cfg': norm_cfg,
            'act_cfg': act_cfg}
        
        self.conv_1_1 = ConvModule(in_channels[0], branch_out_dim, 1, **cfg_dict)
        self.pool_1_2 = nn.AdaptiveMaxPool1d(output_size=1)

        self.conv_2_1 = ConvModule(in_channels[0], branch_out_dim, 3, padding=1, **cfg_dict)
        self.pool_2_2 = nn.AdaptiveMaxPool1d(output_size=1)
        
        self.conv_3_1 = ConvModule(in_channels[0], branch_out_dim, 3, padding=1, **cfg_dict)
        self.conv_3_2 = ConvModule(branch_out_dim, branch_out_dim, 3, padding=1, **cfg_dict)
        self.pool_3_3 = nn.AdaptiveMaxPool1d(output_size=1)

        self.conv_4_1 = ConvModule(in_channels[0], branch_out_dim, 3, padding=1, **cfg_dict)
        self.conv_4_2 = ConvModule(branch_out_dim, branch_out_dim, 3, padding=1, **cfg_dict)
        self.conv_4_3 = ConvModule(branch_out_dim, branch_out_dim, 3, padding=1, **cfg_dict)
        self.pool_4_4 = nn.AdaptiveMaxPool1d(output_size=1)

        self.conv_1 = nn.Linear(branch_out_dim*4, branch_out_dim*2)

        self.conv_5_1 = ConvModule(in_channels[1], branch_out_dim, 1, **cfg_dict)
        self.pool_5_2 = nn.AdaptiveMaxPool1d(output_size=1)

        self.conv_6_1 = ConvModule(in_channels[1], branch_out_dim, 3, padding=1, **cfg_dict)
        self.pool_6_2 = nn.AdaptiveMaxPool1d(output_size=1)
        
        self.conv_7_1 = ConvModule(in_channels[1], branch_out_dim, 3, padding=1, **cfg_dict)
        self.conv_7_2 = ConvModule(branch_out_dim, branch_out_dim, 3, padding=1, **cfg_dict)
        self.pool_7_3 = nn.AdaptiveMaxPool1d(output_size=1)

        self.conv_8_1 = ConvModule(in_channels[1], branch_out_dim, 3, padding=1, **cfg_dict)
        self.conv_8_2 = ConvModule(branch_out_dim, branch_out_dim, 3, padding=1, **cfg_dict)
        self.conv_8_3 = ConvModule(branch_out_dim, branch_out_dim, 3, padding=1, **cfg_dict)
        self.pool_8_4 = nn.AdaptiveMaxPool1d(output_size=1)

        self.conv_2 = nn.Linear(branch_out_dim*4, branch_out_dim*2)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x1, x2):
        x_1 = self.conv_1_1(x1)
        x_1 = self.pool_1_2(x_1).squeeze(-1)

        x_2 = self.conv_2_1(x1)
        x_2 = self.pool_2_2(x_2).squeeze(-1)
        
        x_3 = self.conv_3_1(x1)
        x_3 = self.conv_3_2(x_3)
        x_3 = self.pool_3_3(x_3).squeeze(-1)

        x_4 = self.conv_4_1(x1)
        x_4 = self.conv_4_2(x_4)
        x_4 = self.conv_4_3(x_4)
        x_4 = self.pool_4_4(x_4).squeeze(-1)

        cx_1 = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        cx_1 = self.conv_1(cx_1)
        cx_1 = F.relu6(cx_1)

        x_5 = self.conv_5_1(x2)
        x_5 = self.pool_5_2(x_5).squeeze(-1)

        x_6 = self.conv_6_1(x2)
        x_6 = self.pool_6_2(x_6).squeeze(-1)
        
        x_7 = self.conv_7_1(x2)
        x_7 = self.conv_7_2(x_7)
        x_7 = self.pool_7_3(x_7).squeeze(-1)

        x_8 = self.conv_8_1(x2)
        x_8 = self.conv_8_2(x_8)
        x_8 = self.conv_8_3(x_8)
        x_8 = self.pool_8_4(x_8).squeeze(-1)

        cx_2 = torch.cat([x_5, x_6, x_7, x_8], dim=-1)
        cx_2 = self.conv_2(cx_2)
        cx_2 = F.relu6(cx_2)

        out = torch.cat([cx_1, cx_2], dim=-1)

        return out
