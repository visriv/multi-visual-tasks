import numpy as np
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mvt.utils.init_util import constant_init, kaiming_init
from ..block_builder import HEADS


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.pointwiseconv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.depthwiseconv = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation,
                                       groups=out_channels, bias=bias)

    def forward(self, x):
        x = self.pointwiseconv(x)
        x = nn.ReLU()(x)
        x = self.depthwiseconv(x)
        return x


@HEADS.register_module()
class EfficientLightRPN(nn.Module):
    def __init__(self, 
                 num_class=2,
                 num_input_features=128,
                 rpn_stride=4,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 upsample_strides=[2, 2, 2],
                 num_filters=[256, 256, 256],
                 num_upsample_filters=[256, 256, 256],
                 num_anchor_per_loc=2,
                 box_code_size=7,
                 num_direction_bins=2):
        """
        torch module for region proposal
        :param use_norm: whether using normalization
        :param layer_nums: the list of layer numbers
        :param layer_strides: the list of layer strides
        :param num_filters: the list of filter channels for layers
        :param upsample_strides: the list of strides for up-sampling
        :param num_upsample_filters: the list of filter channels for up-sampling
        :param num_input_filters: the number of input channels
        :param num_anchor_per_loc: number of anchors for each localization
        :param encode_background_as_zeros: whether encoding background as zeros
        :param use_direction_classifier: whether using classifier for direction prediction
        :param box_code_size: the size of encoded box
        :param num_direction_bins: the number of bins for direction classification
        """
        super().__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)

        self._num_input_features = num_input_features
        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters

        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._box_code_size = box_code_size
        self._rpn_stride = rpn_stride

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []
        predictblocks = []
        finalblocks = []

        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(in_filters[i], num_filters[i], layer_num, stride=layer_strides[i])
            blocks.append(block)

        in_deblock_filters = [num_filters[-1]]
        for i in range(len(upsample_strides)-1):
            in_deblock_filters.append(num_upsample_filters[i]+num_filters[-(i+2)])

        for i in range(len(upsample_strides)):
            stride = np.round(upsample_strides[i]).astype(np.int64)
            deblock = nn.Sequential(
                nn.ConvTranspose2d(in_deblock_filters[i], num_upsample_filters[i], stride, stride=stride),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        if len(layer_strides) == len(upsample_strides):
            in_predict_filters = [num_upsample_filters[-1]+num_input_features]
            in_predict_filters.append(num_upsample_filters[-1]*2+num_input_features)
            for i in range(len(upsample_strides) - 2):
                in_predict_filters.append(num_upsample_filters[-(i+2)]*2+num_filters[i])
        else:
            begin_index = len(layer_strides)-len(upsample_strides)
            in_predict_filters = [num_upsample_filters[-1] + num_filters[begin_index-1]]
            for i in range(len(upsample_strides)-1):
                in_predict_filters.append(num_upsample_filters[i+1]*2 + num_filters[i+begin_index-1])

        predict_strides = [1]
        for i in range(len(upsample_strides)-1):
            predict_strides.append(upsample_strides[-(i+1)])
        for i in range(len(upsample_strides)):
            predict_block = nn.Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(in_predict_filters[i], num_upsample_filters[-(i+1)], 3, stride=predict_strides[i], bias=False),
                nn.BatchNorm2d(num_upsample_filters[-(i+1)], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
            predictblocks.append(predict_block)

        for i in range(len(upsample_strides)):
            stride = np.round(np.prod(upsample_strides[:(i+1)])/2.0).astype(np.int64)
            if stride <= 1:
                stride = np.round(stride).astype(np.int64)
                finalblock = nn.Sequential(
                    nn.ZeroPad2d(1),
                    nn.Conv2d(num_upsample_filters[-(i+1)], num_upsample_filters[-(i+1)], 3, stride=stride, bias=False),
                    nn.BatchNorm2d(num_upsample_filters[-(i+1)], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            else:
                finalblock = nn.Sequential(
                    nn.ConvTranspose2d(num_upsample_filters[-(i+1)], num_upsample_filters[-(i+1)], stride, stride=stride),
                    nn.ReLU(),
                )
            finalblocks.append(finalblock)

        self._blocks = nn.ModuleList(blocks)
        self._deblocks = nn.ModuleList(deblocks)
        self._predictblocks = nn.ModuleList(predictblocks)
        self._finalblocks = nn.ModuleList(finalblocks)

        num_cls = num_anchor_per_loc * num_class

        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        self.conv_dir_cls = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * num_direction_bins, 1)

    @property
    def downsample_factor(self):
        factor = self._rpn_stride
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        """
        defining blocks for designing layers
        :param inplanes: the number of input feature channels
        :param planes: the number of output feature channels
        :param num_blocks: number of blocks
        :param stride: the stride of layer
        :return: block and the number of output channels
        """
        block = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01))
            block.add(nn.ReLU())

        return block, planes

    def init_weights(self):
        """Initialize the weights."""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        stage_outputs = []
        ups = []
        predicts = []
        stage_outputs.append(x)
        for i in range(len(self._layer_strides)):
            x = self._blocks[i](x)
            stage_outputs.append(x)

        for i in range(len(self._upsample_strides)):
            x = self._deblocks[i](x)
            x = torch.cat([x, stage_outputs[-i-2]], dim=1)
            ups.append(x)

        for i in range(len(self._upsample_strides)):
            x = self._predictblocks[i](x)
            predicts.append(self._finalblocks[i](x))
            if i < len(self._upsample_strides)-1:
                x = torch.cat([x, ups[-i-1]], dim=1)
            else:
                x = torch.cat(predicts, dim=1)

        box_preds = self.conv_box(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(
            -1, self._num_anchor_per_loc, self._box_code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()

        cls_preds = self.conv_cls(x)
        cls_preds = cls_preds.view(
            -1, self._num_anchor_per_loc, self._num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()

        dir_cls_preds = self.conv_dir_cls(x)
        dir_cls_preds = dir_cls_preds.view(
            -1, self._num_anchor_per_loc, self._num_direction_bins, H, W).permute(0, 1, 3, 4, 2).contiguous()

        ret_dict = {
            "bevrpn_feature_maps": predicts[-1],
            "box_preds": box_preds,
            "cls_preds": cls_preds,
            "dir_cls_preds": dir_cls_preds
        }

        return ret_dict


@HEADS.register_module()
class EfficientRPN(nn.Module):
    """ RPN for object proposals. """
    def __init__(self,
                 num_classes=3,
                 num_input_features=[64, 128, 256],
                 rpn_stride=4,
                 layer_nums=[3, 3, 5, 5],
                 layer_strides=[1, 2, 2, 2],
                 upsample_strides=[2, 2, 2],
                 num_anchor_per_loc=20,
                 box_code_size=7,
                 num_direction_bins=2,
                 is_multi=True):
        """
        torch module for region proposal
        :param use_norm: whether using normalization
        :param layer_nums: the list of layer numbers
        :param layer_strides: the list of layer strides
        :param num_filters: the list of filter channels for layers
        :param upsample_strides: the list of strides for up-sampling
        :param num_upsample_filters: the list of filter channels for up-sampling
        :param num_input_filters: the number of input channels
        :param num_anchor_per_loc: number of anchors for each localization
        :param encode_background_as_zeros: whether encoding background as zeros
        :param use_direction_classifier: whether using classifier for direction prediction
        :param box_code_size: the size of encoded box
        :param num_direction_bins: the number of bins for direction classification
        """
        super(EfficientRPN, self).__init__()

        self._num_input_features = num_input_features
        self._layer_strides = layer_strides
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_classes = num_classes
        self._box_code_size = box_code_size
        self._rpn_stride = rpn_stride
        self._is_multi = is_multi

        blocks = []
        downblocks = []
        upblocks = []
        upaddblocks = []
        predblocks = []

        if self._is_multi:
            if layer_strides[0] == 1:
                block_filters = num_input_features + [num_input_features[-1]*2]
            else:
                block_filters = num_input_features + [num_input_features[-1]*2, num_input_features[-1]*4]
        else:
            block_filters = []
            for i in range(len(layer_strides)):
                if layer_strides[0] == 1:
                    block_filters.append(num_input_features * (2 ** i))
                else:
                    if i == 0:
                        block_filters.append(num_input_features * (2 ** i))
                    block_filters.append(num_input_features * (2 ** (i + 1)))

        for i, layer_num in enumerate(layer_nums):
            if layer_strides[0] == 1:
                if i == 0:
                    downblocks.append(nn.Sequential(
                        nn.ZeroPad2d(1),
                        nn.Conv2d(block_filters[i], block_filters[i], 3, stride=layer_strides[i], bias=False),
                        nn.BatchNorm2d(block_filters[i], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ))
                else:
                    downblocks.append(nn.Sequential(
                        nn.ZeroPad2d(1),
                        nn.Conv2d(block_filters[i-1], block_filters[i], 3, stride=layer_strides[i], bias=False),
                        nn.BatchNorm2d(block_filters[i], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ))

                block = self._make_layer(block_filters[i], layer_num)
            else:
                downblocks.append(nn.Sequential(
                    nn.ZeroPad2d(1),
                    nn.Conv2d(block_filters[i], block_filters[i+1], 3, stride=layer_strides[i], bias=False),
                    nn.BatchNorm2d(block_filters[i+1], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ))

                block = self._make_layer(block_filters[i+1], layer_num)
            blocks.append(block)

        for i in range(len(upsample_strides)):
            stride = np.round(upsample_strides[i]).astype(np.int64)
            upblocks.append(nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=True),
                nn.ZeroPad2d(1),
                nn.Conv2d(block_filters[-(i+1)], block_filters[-(i+2)], 3, stride=1, bias=False),
                nn.BatchNorm2d(block_filters[-(i+2)], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
            upaddblocks.append(nn.Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(block_filters[-(i+2)], block_filters[-(i+2)], 3, stride=1, bias=False),
                nn.BatchNorm2d(block_filters[-(i+2)], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        if layer_strides[0] == 1:
            t = 0
        else:
            t = 1

        predblocks.append(nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(block_filters[t], block_filters[t], 3, stride=1, bias=False),
            nn.BatchNorm2d(block_filters[t], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        ))

        predblocks.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(block_filters[t+1], block_filters[t], 3, stride=1, bias=False),
            nn.BatchNorm2d(block_filters[t], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        ))

        predblocks.append(nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(block_filters[t+2], block_filters[t], 3, stride=1, bias=False),
            nn.BatchNorm2d(block_filters[t], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        ))

        self._blocks = nn.ModuleList(blocks)
        self._downblocks = nn.ModuleList(downblocks)
        self._upblocks = nn.ModuleList(upblocks)
        self._upaddblocks = nn.ModuleList(upaddblocks)
        self._predblocks = nn.ModuleList(predblocks)

        self._conv_cls = nn.Conv2d(block_filters[t] * 3, num_anchor_per_loc*num_classes, 1)
        self._conv_box = nn.Conv2d(block_filters[t] * 3, num_anchor_per_loc*box_code_size, 1)
        self._conv_dir = nn.Conv2d(block_filters[t] * 3, num_anchor_per_loc*num_direction_bins, 1)

    def _make_layer(self, planes, num_blocks):
        """
        defining blocks for designing layers
        :param planes: the number of output feature channels
        :param num_blocks: number of blocks
        :return: block
        """
        block = nn.Sequential()
        for j in range(num_blocks):
            block.add(SeparableConv2d(planes, planes, 3, padding=1, bias=False))
            block.add(nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01))
            block.add(nn.ReLU())

        return block

    def init_weights(self):
        """Initialize the weights."""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, in_features):
        downs = []
        ups = []

        len_scale = len(self._layer_strides)
        len_up = len(self._upsample_strides)

        if self._is_multi:
            x = in_features[0]
        else:
            x = in_features
        for i in range(len_scale):
            if self._is_multi and (i == 1 or i == 2) and self._layer_strides[0] == 2:
                x = torch.add(x, in_features[i])
            x = self._downblocks[i](x)
            if self._is_multi and (i == 1 or i == 2) and self._layer_strides[0] == 1:
                x = torch.add(x, in_features[i])
            x = self._blocks[i](x)
            downs.append(x)

        for i in range(len_up):
            x = self._upblocks[i](x)
            x = torch.add(x, downs[-(i+2)])
            x = self._upaddblocks[i](x)
            ups.append(x)

        preds = []
        for i in range(len_up):
            x = self._predblocks[-(i+1)](torch.add(ups[i], downs[-(i+2)]))
            preds.append(x)
        x = torch.cat(preds, dim=-3)

        box_preds = self._conv_box(x)
        cls_preds = self._conv_cls(x)
        dir_preds = self._conv_dir(x)

        C, H, W = box_preds.shape[1:]

        box_preds = box_preds.view(-1, self._num_anchor_per_loc, self._box_code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc, self._num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
        dir_preds = dir_preds.view(-1, self._num_anchor_per_loc, self._num_direction_bins, H, W).permute(0, 1, 3, 4, 2).contiguous()

        ret_dict = {
            "bevrpn_feature_maps": x,
            "box_preds": box_preds,
            "cls_preds": cls_preds,
            "dir_cls_preds": dir_preds
        }

        return ret_dict
