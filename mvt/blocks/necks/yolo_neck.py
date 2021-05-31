# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ..block_builder import NECKS
from mvt.cores.ops import ConvModule
from mvt.cores.layer_ops import brick as vn_layer


class DetectionBlock(nn.Module):
    """Detection block in YOLO neck.
    Let out_channels = n, the DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
        1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n.
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(DetectionBlock, self).__init__()
        double_out_channels = out_channels * 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels, out_channels, 1, **cfg)
        self.conv2 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv3 = ConvModule(double_out_channels, out_channels, 1, **cfg)
        self.conv4 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv5 = ConvModule(double_out_channels, out_channels, 1, **cfg)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        out = self.conv5(tmp)
        return out


@NECKS.register_module()
class YOLOV3Neck(nn.Module):
    """The neck of YOLOV3.

    It can be treated as a simplified version of FPN. It
    will take the result from Darknet backbone and do some upsampling and
    concatenation. It will finally output the detection result.

    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV3Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(YOLOV3Neck, self).__init__()
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        self.detect1 = DetectionBlock(in_channels[0], out_channels[0], **cfg)
        for i in range(1, self.num_scales):
            in_c, out_c = self.in_channels[i], self.out_channels[i]
            self.add_module(f'conv{i}', ConvModule(in_c, out_c, 1, **cfg))
            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f'detect{i+1}',
                            DetectionBlock(in_c + out_c, out_c, **cfg))

    def forward(self, feats):
        assert len(feats) == self.num_scales

        # processed from bottom (high-lvl) to top (low-lvl)
        outs = []
        out = self.detect1(feats[-1])
        outs.append(out)

        for i, x in enumerate(reversed(feats[:-1])):
            conv = getattr(self, f'conv{i+1}')
            tmp = conv(out)

            # Cat with low-lvl feats
            tmp = F.interpolate(tmp, scale_factor=2)
            tmp = torch.cat((tmp, x), 1)

            detect = getattr(self, f'detect{i+2}')
            out = detect(tmp)
            outs.append(out)

        return tuple(outs)

    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in ConvModule
        pass


@NECKS.register_module()
class DarkNeck(nn.Module):
    def __init__(self, input_channels=32):
        super(DarkNeck, self).__init__()
        layer_list = [
            # the following is extra
            # layer 3
            # output third scale, largest
            OrderedDict([
                ('head_body_1', vn_layer.HeadBody(input_channels * (2 ** 5), first_head=True)),
            ]),

            # layer 4
            OrderedDict([
                ('trans_1', vn_layer.Transition(input_channels * (2 ** 4))),
            ]),

            # layer 5
            # output second scale
            OrderedDict([
                ('head_body_2', vn_layer.HeadBody(input_channels * (2 ** 4 + 2 ** 3))),
            ]),

            # layer 6
            OrderedDict([
                ('trans_2', vn_layer.Transition(input_channels * (2 ** 3))),
            ]),

            # layer 7
            # output first scale, smallest
            OrderedDict([
                ('head_body_3', vn_layer.HeadBody(input_channels * (2 ** 3 + 2 ** 2))),
            ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        stage_6, stage_5, stage_4 = x
        head_body_1 = self.layers[0](stage_6)
        trans_1 = self.layers[1](head_body_1)

        concat_2 = torch.cat([trans_1, stage_5], 1)
        head_body_2 = self.layers[2](concat_2)
        trans_2 = self.layers[3](head_body_2)

        concat_3 = torch.cat([trans_2, stage_4], 1)
        head_body_3 = self.layers[4](concat_3)

        # stage 6, stage 5, stage 4
        features = [head_body_1, head_body_2, head_body_3]
        return features


# PAN+SPP
@NECKS.register_module()
class YoloV4DarkNeck(nn.Module):
    def __init__(self):
        super(YoloV4DarkNeck, self).__init__()
        layer_list = [
            OrderedDict([
                ('head_body0_0', vn_layer.MakeNConv([512, 1024], 1024, 3)),
                ('spp', vn_layer.SpatialPyramidPooling()),
                ('head_body0_1', vn_layer.MakeNConv([512, 1024], 2048, 3)), ]
            ),
            OrderedDict([
                ('trans_0', vn_layer.FuseStage(512)),
                ('head_body1_0', vn_layer.MakeNConv([256, 512], 512, 5))]
            ),

            OrderedDict([
                ('trans_1', vn_layer.FuseStage(256)),
                ('head_body2_1', vn_layer.MakeNConv([128, 256], 256, 5))
            ]),  # out0

            OrderedDict([
                ('trans_2', vn_layer.FuseStage(128, is_reversal=True)),
                ('head_body1_1', vn_layer.MakeNConv([256, 512], 512, 5))]
            ),  # out1

            OrderedDict([
                ('trans_3', vn_layer.FuseStage(256, is_reversal=True)),
                ('head_body0_2', vn_layer.MakeNConv([512, 1024], 1024, 5))]
            ),  # out2
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        out3, out4, out5 = x
        out5 = self.layers[0](out5)
        out4 = self.layers[1]([out4, out5])

        out3 = self.layers[2]([out3, out4])  # out0 large
        out4 = self.layers[3]([out3, out4])  # out1
        out5 = self.layers[4]([out4, out5])  # out2 small

        return [out5, out4, out3]

    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in ConvModule
        pass
