# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import logging
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from .resnet import ResBlock
from ..block_builder import BACKBONES
from model.mvt.cores.ops import ConvModule
from model.mvt.utils.init_util import constant_init, kaiming_init
from model.mvt.utils.checkpoint_util import load_checkpoint
from model.mvt.cores.layer_ops import brick as vn_layer


@BACKBONES.register_module()
class Darknet(nn.Module):
    """Darknet backbone.
    Args:
        depth (int): Depth of Darknet. Currently only support 53.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
    Example:
        >>> import torch
        >>> self = Darknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """

    # Dict(depth: (layers, channels))
    arch_settings = {
        53: (
            (1, 2, 8, 8, 4),
            ((32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)),
        )
    }

    def __init__(
        self,
        depth=53,
        out_indices=(3, 4, 5),
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_cfg=dict(type="LeakyReLU", negative_slope=0.1),
        norm_eval=True,
    ):
        super(Darknet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for darknet")
        self.depth = depth
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.layers, self.channels = self.arch_settings[depth]

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(3, 32, 3, padding=1, **cfg)

        self.cr_blocks = ["conv1"]
        for i, n_layers in enumerate(self.layers):
            layer_name = f"conv_res_block{i + 1}"
            in_c, out_c = self.channels[i]
            self.add_module(
                layer_name, self.make_conv_res_block(in_c, out_c, n_layers, **cfg)
            )
            self.cr_blocks.append(layer_name)

        self.norm_eval = norm_eval

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError("pretrained must be a str or None")

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self, self.cr_blocks[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(Darknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    @staticmethod
    def make_conv_res_block(
        in_channels,
        out_channels,
        res_repeat,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_cfg=dict(type="LeakyReLU", negative_slope=0.1),
    ):
        """In Darknet backbone, ConvLayer is usually followed by ResBlock. This
        function will make that. The Conv layers always have 3x3 filters with
        stride=2. The number of the filters in Conv layer is the same as the
        out channels of the ResBlock.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            res_repeat (int): The number of ResBlocks.
            conv_cfg (dict): Config dict for convolution layer. Default: None.
            norm_cfg (dict): Dictionary to construct and config norm layer.
                Default: dict(type='BN', requires_grad=True)
            act_cfg (dict): Config dict for activation layer.
                Default: dict(type='LeakyReLU', negative_slope=0.1).
        """

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        model = nn.Sequential()
        model.add_module(
            "conv", ConvModule(in_channels, out_channels, 3, stride=2, padding=1, **cfg)
        )
        for idx in range(res_repeat):
            model.add_module("res{}".format(idx), ResBlock(out_channels, **cfg))
        return model


@BACKBONES.register_module()
class CSPDarknet53(nn.Module):
    custom_layers = (vn_layer.Resblock_body, vn_layer.Resblock_body.custom_layers)

    def __init__(self, layers=(1, 2, 8, 8, 4), input_channels=32, pretrained=None):
        super().__init__()
        self.inplanes = input_channels
        self.conv1 = vn_layer.Conv2dBatchMish(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList(
            [
                vn_layer.Resblock_body(
                    self.inplanes, self.feature_channels[0], layers[0], first=True
                ),
                vn_layer.Resblock_body(
                    self.feature_channels[0],
                    self.feature_channels[1],
                    layers[1],
                    first=False,
                ),
                vn_layer.Resblock_body(
                    self.feature_channels[1],
                    self.feature_channels[2],
                    layers[2],
                    first=False,
                ),
                vn_layer.Resblock_body(
                    self.feature_channels[2],
                    self.feature_channels[3],
                    layers[3],
                    first=False,
                ),
                vn_layer.Resblock_body(
                    self.feature_channels[3],
                    self.feature_channels[4],
                    layers[4],
                    first=False,
                ),
            ]
        )

        self.init_weights(pretrained)

    def __modules_recurse(self, mod=None):
        """This function will recursively loop over all module children.
        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self
        for module in mod.children():
            if isinstance(
                module, (nn.ModuleList, nn.Sequential, CSPDarknet53.custom_layers)
            ):
                yield from self.__modules_recurse(module)
            else:
                yield module

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            weights = vn_layer.WeightLoader(pretrained)
            for module in self.__modules_recurse():
                try:
                    weights.load_layer(module)
                    print(f"Layer loaded: {module}")
                    if weights.start >= weights.size:
                        print(
                            f"Finished loading weights [{weights.start}/{weights.size} weights]"
                        )
                        break
                except NotImplementedError:
                    print(f"Layer skipped: {module.__class__.__name__}")
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return [out3, out4, out5]
