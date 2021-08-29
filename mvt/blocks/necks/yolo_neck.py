# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ..block_builder import NECKS
from mvt.cores.ops import ConvModule
from mvt.utils.init_util import xavier_init
from mvt.cores.layer_ops import brick as vn_layer
from mvt.blocks.backbones.darknetcsp import BottleneckCSP, BottleneckCSP2, Conv


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

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_cfg=dict(type="LeakyReLU", negative_slope=0.1),
    ):
        super(DetectionBlock, self).__init__()
        double_out_channels = out_channels * 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels, out_channels, 1, **cfg)
        self.conv2 = ConvModule(out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv3 = ConvModule(double_out_channels, out_channels, 1, **cfg)
        self.conv4 = ConvModule(out_channels, double_out_channels, 3, padding=1, **cfg)
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

    def __init__(
        self,
        num_scales,
        in_channels,
        out_channels,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_cfg=dict(type="LeakyReLU", negative_slope=0.1),
    ):
        super(YOLOV3Neck, self).__init__()
        assert num_scales == len(in_channels) == len(out_channels)
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
            self.add_module(f"conv{i}", ConvModule(in_c, out_c, 1, **cfg))
            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f"detect{i+1}", DetectionBlock(in_c + out_c, out_c, **cfg))

    def forward(self, feats):
        assert len(feats) == self.num_scales

        # processed from bottom (high-lvl) to top (low-lvl)
        outs = []
        out = self.detect1(feats[-1])
        outs.append(out)

        for i, x in enumerate(reversed(feats[:-1])):
            conv = getattr(self, f"conv{i+1}")
            tmp = conv(out)

            # Cat with low-lvl feats
            tmp = F.interpolate(tmp, scale_factor=2)
            tmp = torch.cat((tmp, x), 1)

            detect = getattr(self, f"detect{i+2}")
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
            OrderedDict(
                [
                    (
                        "head_body_1",
                        vn_layer.HeadBody(input_channels * (2 ** 5), first_head=True),
                    ),
                ]
            ),
            # layer 4
            OrderedDict(
                [
                    ("trans_1", vn_layer.Transition(input_channels * (2 ** 4))),
                ]
            ),
            # layer 5
            # output second scale
            OrderedDict(
                [
                    (
                        "head_body_2",
                        vn_layer.HeadBody(input_channels * (2 ** 4 + 2 ** 3)),
                    ),
                ]
            ),
            # layer 6
            OrderedDict(
                [
                    ("trans_2", vn_layer.Transition(input_channels * (2 ** 3))),
                ]
            ),
            # layer 7
            # output first scale, smallest
            OrderedDict(
                [
                    (
                        "head_body_3",
                        vn_layer.HeadBody(input_channels * (2 ** 3 + 2 ** 2)),
                    ),
                ]
            ),
        ]
        self.layers = nn.ModuleList(
            [nn.Sequential(layer_dict) for layer_dict in layer_list]
        )
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
            OrderedDict(
                [
                    ("head_body0_0", vn_layer.MakeNConv([512, 1024], 1024, 3)),
                    ("spp", vn_layer.SpatialPyramidPooling()),
                    ("head_body0_1", vn_layer.MakeNConv([512, 1024], 2048, 3)),
                ]
            ),
            OrderedDict(
                [
                    ("trans_0", vn_layer.FuseStage(512)),
                    ("head_body1_0", vn_layer.MakeNConv([256, 512], 512, 5)),
                ]
            ),
            OrderedDict(
                [
                    ("trans_1", vn_layer.FuseStage(256)),
                    ("head_body2_1", vn_layer.MakeNConv([128, 256], 256, 5)),
                ]
            ),  # out0
            OrderedDict(
                [
                    ("trans_2", vn_layer.FuseStage(128, is_reversal=True)),
                    ("head_body1_1", vn_layer.MakeNConv([256, 512], 512, 5)),
                ]
            ),  # out1
            OrderedDict(
                [
                    ("trans_3", vn_layer.FuseStage(256, is_reversal=True)),
                    ("head_body0_2", vn_layer.MakeNConv([512, 1024], 1024, 5)),
                ]
            ),  # out2
        ]

        self.layers = nn.ModuleList(
            [nn.Sequential(layer_dict) for layer_dict in layer_list]
        )

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


@NECKS.register_module()
class YOLOV4Neck(nn.Module):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int or List[int]): Number of output channels (used at
            each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs=None,
        csp_repetition=3,
        start_level=0,
        end_level=-1,
        norm_cfg=dict(type="BN", requires_grad=True, eps=0.001, momentum=0.03),
        act_cfg=dict(type="Mish"),
        csp_act_cfg=dict(type="Mish"),
        upsample_cfg=dict(mode="nearest"),
    ):

        super(YOLOV4Neck, self).__init__()

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        if isinstance(out_channels, list):
            self.out_channels = out_channels
            num_outs = len(out_channels)
        else:
            assert num_outs is not None
            self.out_channels = [out_channels] * num_outs

        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs == self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        cfg = dict(norm_cfg=norm_cfg, act_cfg=act_cfg, csp_act_cfg=csp_act_cfg)

        # 1x1 convs to shrink channels count before upsample and concat
        self.pre_upsample_convs = nn.ModuleList()

        # 1x1 convs to shrink backbone output channels count before concat
        self.backbone_pre_concat_convs = nn.ModuleList()

        # CSP convs to shrink channels after concat
        self.post_upsample_concat_csp = nn.ModuleList()

        # strided convs used to downsample
        self.downsample_convs = nn.ModuleList()

        # CSP convs after downsample
        self.post_downsample_concat_csp = nn.ModuleList()

        # yolov4 use 3x3 convs to process the final output
        self.out_convs = nn.ModuleList()

        # top-down path
        # from top level(smaller and deeper heat maps)
        # to bottom level(bigger and shallower heat maps) input index
        # starts with the topmost output of the backbone
        current_channels = in_channels[self.backbone_end_level - 1]
        to_bottom_up_concat_channels = []
        for i in range(self.backbone_end_level - 1, self.start_level, -1):
            bottom_channels = in_channels[i - 1]
            # yolov4 style
            target_channels = bottom_channels // 2

            # yolov4 send the input of this 1x1 conv to bottom up process flow
            # for concatenation
            to_bottom_up_concat_channels.append(current_channels)
            pre_up_conv = Conv(
                in_channels=current_channels,
                out_channels=target_channels,
                kernel_size=1,
                **cfg,
            )

            backbone_pre_cat_conv = Conv(
                in_channels=bottom_channels,
                out_channels=target_channels,
                kernel_size=1,
                **cfg,
            )

            post_upcat_csp = BottleneckCSP2(
                in_channels=2 * target_channels,
                # channel count doubles after concatenation
                out_channels=target_channels,
                repetition=csp_repetition,
                shortcut=False,
                **cfg,
            )
            self.pre_upsample_convs.insert(0, pre_up_conv)
            self.backbone_pre_concat_convs.insert(0, backbone_pre_cat_conv)
            self.post_upsample_concat_csp.insert(0, post_upcat_csp)
            current_channels = target_channels

        # bottom-up path
        # from bottom level(bigger and shallower heat maps)
        # to top level(smaller and deeper heat maps)
        to_output_channels = [current_channels]
        for i in range(self.start_level, self.backbone_end_level - 1):
            top_channels = to_bottom_up_concat_channels.pop(-1)

            down_conv = Conv(
                in_channels=current_channels,
                out_channels=top_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                **cfg,
            )

            post_downcat_csp = BottleneckCSP2(
                in_channels=2 * top_channels,
                # channel count doubles after concatenation
                out_channels=top_channels,
                repetition=csp_repetition,
                shortcut=False,
                **cfg,
            )
            self.downsample_convs.append(down_conv)
            self.post_downsample_concat_csp.append(post_downcat_csp)
            to_output_channels.append(top_channels)
            current_channels = top_channels

        # build output conv
        for i in range(num_outs):
            before_conv_channels = to_output_channels[i]
            out_channels = self.out_channels[i]
            out_conv = Conv(
                in_channels=before_conv_channels,
                out_channels=out_channels,
                kernel_size=3,
                **cfg,
            )
            self.out_convs.append(out_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        used_backbone_levels = self.backbone_end_level - self.start_level

        # build top-down path
        x = inputs[self.backbone_end_level - 1]
        bottom_up_merge = []

        for i in range(used_backbone_levels - 1, 0, -1):  # [2, 1]
            pre_up_conv = self.pre_upsample_convs[i - 1]
            backbone_pre_cat_conv = self.backbone_pre_concat_convs[i - 1]
            post_upcat_csp = self.post_upsample_concat_csp[i - 1]

            inputs_bottom = backbone_pre_cat_conv(inputs[self.start_level + i - 1])

            # yolov4 send the input of this 1x1 conv to bottom up process flow
            # for concatenation
            bottom_up_merge.append(x)
            x = pre_up_conv(x)

            if "scale_factor" in self.upsample_cfg:
                x = F.interpolate(x, **self.upsample_cfg)
            else:
                bottom_shape = inputs_bottom.shape[2:]
                x = F.interpolate(x, size=bottom_shape, **self.upsample_cfg)

            x = torch.cat((inputs_bottom, x), dim=1)
            x = post_upcat_csp(x)

        # build additional bottom up path

        outs = [x]
        for i in range(self.backbone_end_level - self.start_level - 1):
            down_conv = self.downsample_convs[i]
            post_downcat_csp = self.post_downsample_concat_csp[i]
            x = down_conv(x)
            x = torch.cat((x, bottom_up_merge.pop(-1)), dim=1)
            x = post_downcat_csp(x)
            outs.append(x)

        # yolov4 use 3x3 convs to process the final output

        for i in range(len(outs)):
            outs[i] = self.out_convs[i](outs[i])

        return tuple(outs)


@NECKS.register_module()
class YOLOV5Neck(nn.Module):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int or List[int]): Number of output channels (used at
            each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs=None,
        csp_repetition=3,
        start_level=0,
        end_level=-1,
        norm_cfg=dict(type="BN", requires_grad=True, eps=0.001, momentum=0.03),
        act_cfg=dict(type="Mish"),
        csp_act_cfg=dict(type="Mish"),
        upsample_cfg=dict(mode="nearest"),
    ):

        super(YOLOV5Neck, self).__init__()

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        if isinstance(out_channels, list):
            self.out_channels = out_channels
            num_outs = len(out_channels)
        else:
            assert num_outs is not None
            self.out_channels = [out_channels] * num_outs

        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs == self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        cfg = dict(norm_cfg=norm_cfg, act_cfg=act_cfg, csp_act_cfg=csp_act_cfg)

        # shrink channels count before upsample and concat
        self.pre_upsample_convs = nn.ModuleList()

        # yolov5 has no 1x1 conv before feeding the output of the backbone
        # to top down process flow for concatenation

        # CSP convs to shrink channels after concat
        self.post_upsample_concat_csp = nn.ModuleList()

        # convs for downsample
        self.downsample_convs = nn.ModuleList()

        # CSP convs after downsample
        self.post_downsample_concat_csp = nn.ModuleList()

        # yolov5 has no final 1x1 conv to process the final output

        # top-down path
        # from top level(smaller and deeper heat maps)
        # to bottom level(bigger and shallower heat maps) input index
        # starts with the topmost output of the backbone
        current_channels = in_channels[self.backbone_end_level - 1]
        to_bottom_up_concat_channels = []
        for i in range(self.backbone_end_level - 1, self.start_level, -1):
            bottom_channels = in_channels[i - 1]
            # yolov5 style
            target_channels = bottom_channels

            # yolov5 send the output of this 1x1 conv to bottom up process flow
            # for concatenation
            pre_up_conv = Conv(
                in_channels=current_channels,
                out_channels=target_channels,
                kernel_size=1,
                **cfg,
            )
            to_bottom_up_concat_channels.append(target_channels)

            post_upcat_csp = BottleneckCSP(
                in_channels=2 * target_channels,
                # channel count doubles after concatenation
                out_channels=target_channels,
                repetition=csp_repetition,
                shortcut=False,
                **cfg,
            )
            self.pre_upsample_convs.insert(0, pre_up_conv)
            self.post_upsample_concat_csp.insert(0, post_upcat_csp)

            current_channels = target_channels

        # bottom-up path
        # from bottom level(bigger and shallower heat maps)
        # to top level(smaller and deeper heat maps)
        to_output_channels = [current_channels]
        for i in range(self.start_level, self.backbone_end_level - 1):
            top_channels = to_bottom_up_concat_channels.pop(-1)
            # yolov5 style
            target_channels = self.out_channels[i - self.start_level + 1]

            down_conv = Conv(
                in_channels=current_channels,
                out_channels=top_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                **cfg,
            )

            post_downcat_csp = BottleneckCSP(
                in_channels=2 * top_channels,
                out_channels=target_channels,
                repetition=csp_repetition,
                shortcut=False,
                **cfg,
            )
            self.downsample_convs.append(down_conv)
            self.post_downsample_concat_csp.append(post_downcat_csp)
            to_output_channels.append(top_channels)
            current_channels = target_channels
        # yolov5 has no output 1x1 conv

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        used_backbone_levels = self.backbone_end_level - self.start_level

        # build top-down path
        x = inputs[self.backbone_end_level - 1]
        bottom_up_merge = []

        for i in range(used_backbone_levels - 1, 0, -1):  # [2, 1]
            pre_up_conv = self.pre_upsample_convs[i - 1]
            post_upcat_csp = self.post_upsample_concat_csp[i - 1]

            # yolov5 has no 1x1 conv before feeding the output of the backbone
            # to top down process flow for concatenation
            inputs_bottom = inputs[self.start_level + i - 1]

            # yolov5 send the output of this 1x1 conv to bottom up process flow
            # for concatenation
            x = pre_up_conv(x)
            bottom_up_merge.append(x)

            if "scale_factor" in self.upsample_cfg:
                x = F.interpolate(x, **self.upsample_cfg)
            else:
                bottom_shape = inputs_bottom.shape[2:]
                x = F.interpolate(x, size=bottom_shape, **self.upsample_cfg)

            x = torch.cat((inputs_bottom, x), dim=1)
            x = post_upcat_csp(x)

        # build additional bottom up path

        outs = [x]
        for i in range(self.backbone_end_level - self.start_level - 1):
            down_conv = self.downsample_convs[i]
            post_downcat_csp = self.post_downsample_concat_csp[i]
            x = down_conv(x)
            x = torch.cat((x, bottom_up_merge.pop(-1)), dim=1)
            x = post_downcat_csp(x)
            outs.append(x)

        # yolov5 has no final 3x3 conv to process the final output

        return tuple(outs)
