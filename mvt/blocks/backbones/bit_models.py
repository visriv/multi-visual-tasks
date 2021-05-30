# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""

from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import math
import logging
import torch.nn as nn
import torch.nn.functional as F

# mtl relevants
from mtl.utils.checkpoint_util import load_checkpoint
from mtl.utils.init_util import constant_init, kaiming_init
from ..block_builder import BACKBONES

class StdConv2d(nn.Conv2d):

  def forward(self, x):
    w = self.weight
    v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    w = (w - m) / torch.sqrt(v + 1e-10)
    return F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
  return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                   padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
  return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                   padding=0, bias=bias)


def tf2th(conv_weights):
  """Possibly convert HWIO to OIHW."""
  if conv_weights.ndim == 4:
    conv_weights = conv_weights.transpose([3, 2, 0, 1])
  return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
  """Pre-activation (v2) bottleneck block.

  Follows the implementation of "Identity Mappings in Deep Residual Networks":
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

  Except it puts the stride on 3x3 conv when available.
  """

  def __init__(self, cin, cout=None, cmid=None, stride=1):
    super().__init__()
    cout = cout or cin
    cmid = cmid or cout//4

    self.gn1 = nn.GroupNorm(32, cin)
    self.conv1 = conv1x1(cin, cmid)
    self.gn2 = nn.GroupNorm(32, cmid)
    self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
    self.gn3 = nn.GroupNorm(32, cmid)
    self.conv3 = conv1x1(cmid, cout)
    self.relu = nn.ReLU(inplace=True)

    if (stride != 1 or cin != cout):
      # Projection also with pre-activation according to paper.
      self.downsample = conv1x1(cin, cout, stride)

  def forward(self, x):
    out = self.relu(self.gn1(x))

    # Residual branch
    residual = x
    if hasattr(self, 'downsample'):
      residual = self.downsample(out)

    # Unit's branch
    out = self.conv1(out)
    out = self.conv2(self.relu(self.gn2(out)))
    out = self.conv3(self.relu(self.gn3(out)))

    return out + residual

  def load_from(self, weights, prefix=''):
    convname = 'standardized_conv2d'
    with torch.no_grad():
      self.conv1.weight.copy_(tf2th(weights['{}a/{}/kernel'.format(prefix, convname)]))
      self.conv2.weight.copy_(tf2th(weights['{}b/{}/kernel'.format(prefix, convname)]))
      self.conv3.weight.copy_(tf2th(weights['{}c/{}/kernel'.format(prefix, convname)]))
      self.gn1.weight.copy_(tf2th(weights['{}a/group_norm/gamma'.format(prefix)]))
      self.gn2.weight.copy_(tf2th(weights['{}b/group_norm/gamma'.format(prefix)]))
      self.gn3.weight.copy_(tf2th(weights['{}c/group_norm/gamma'.format(prefix)]))
      self.gn1.bias.copy_(tf2th(weights['{}a/group_norm/beta'.format(prefix)]))
      self.gn2.bias.copy_(tf2th(weights['{}b/group_norm/beta'.format(prefix)]))
      self.gn3.bias.copy_(tf2th(weights['{}c/group_norm/beta'.format(prefix)]))
      if hasattr(self, 'downsample'):
        w = weights['{}a/proj/{}/kernel'.format(prefix, convname)]
        self.downsample.weight.copy_(tf2th(w))

  def load_for_finetune(self, weights, prefix=''):
    convname = 'standardized_conv2d'
    nn.init.constant(self.conv1.weight, tf2th(weights['{}a/{}/kernel'.format(prefix, convname)]))
    nn.init.constant(self.conv2.weight, weights['{}b/{}/kernel'.format(prefix, convname)])
    nn.init.constant(self.conv3.weight, weights['{}c/{}/kernel'.format(prefix, convname)])
    nn.init.constant(self.gn1.weight, weights['{}a/group_norm/gamma'.format(prefix)])
    nn.init.constant(self.gn2.weight, weights['{}b/group_norm/gamma'.format(prefix)])
    nn.init.constant(self.gn3.weight, weights['{}c/group_norm/gamma'.format(prefix)])
    nn.init.constant(self.gn1.bias, weights['{}a/group_norm/beta'.format(prefix)])
    nn.init.constant(self.gn2.bias, weights['{}b/group_norm/beta'.format(prefix)])
    nn.init.constant(self.gn3.bias, weights['{}c/group_norm/beta'.format(prefix)])
    # self.conv1.weight.copy_(tf2th(weights[f'{prefix}a/{convname}/kernel']))
    # self.conv2.weight.copy_(tf2th(weights[f'{prefix}b/{convname}/kernel']))
    # self.conv3.weight.copy_(tf2th(weights[f'{prefix}c/{convname}/kernel']))
    # self.gn1.weight.copy_(tf2th(weights[f'{prefix}a/group_norm/gamma']))
    # self.gn2.weight.copy_(tf2th(weights[f'{prefix}b/group_norm/gamma']))
    # self.gn3.weight.copy_(tf2th(weights[f'{prefix}c/group_norm/gamma']))
    # self.gn1.bias.copy_(tf2th(weights[f'{prefix}a/group_norm/beta']))
    # self.gn2.bias.copy_(tf2th(weights[f'{prefix}b/group_norm/beta']))
    # self.gn3.bias.copy_(tf2th(weights[f'{prefix}c/group_norm/beta']))
    if hasattr(self, 'downsample'):
      w = weights['{}a/proj/{}/kernel'.format(prefix, convname)]
      nn.init.constant(self.downsample.weight, w)
      # self.downsample.weight.copy_(tf2th(w))


@BACKBONES.register_module()
class ResNetV2(nn.Module):
  """Implementation of Pre-activation (v2) ResNet mode."""

  def __init__(self, block_units, width_factor, head_size=21843, zero_head=False, hash_bit=1024, float_bit=512,is_training=True):
    super().__init__()
    wf = width_factor  # shortcut 'cause we'll use it a lot.

    # The following will be unreadable if we split lines.
    # pylint: disable=line-too-long
    self.init_scale = 1.0
    self.scale = self.init_scale
    self.power = 0.5
    self.gamma = 0.005
    self.iter_num = 0
    self.step_size = 200
    self.training = is_training
    self.activation = nn.Tanh()
    self.root = nn.Sequential(OrderedDict([
        ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
        ('pad', nn.ConstantPad2d(1, 0)),
        ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
        # The following is subtly not the same!
        # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))

    self.body = nn.Sequential(OrderedDict([
        ('block1', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf)) for i in range(2, block_units[0] + 1)],
        ))),
        ('block2', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf)) for i in range(2, block_units[1] + 1)],
        ))),
        ('block3', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf)) for i in range(2, block_units[2] + 1)],
        ))),
        ('block4', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf)) for i in range(2, block_units[3] + 1)],
        ))),
    ]))
    # pylint: enable=line-too-long

    self.zero_head = zero_head
    self.head = nn.Sequential(OrderedDict([
        ('gn', nn.GroupNorm(32, 2048*wf)),
        ('relu', nn.ReLU(inplace=True)),
        ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
    ]))
        # ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
        # ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),

  def forward(self, x):
    x = self.body(self.root(x))
    x = self.head(x)
    assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
    return x[...,0,0]

  def init_weights(self, pretrained=True):
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
      raise TypeError('pretrained must be a str or None')

  def load_from(self, weights, prefix='resnet/'):
    with torch.no_grad():
      for bname, block in self.body.named_children():
        for uname, unit in block.named_children():
          unit.load_from(weights, prefix='{}{}/{}/'.format(prefix, bname, uname))
      self.root.conv.weight.copy_(tf2th(weights['{}root_block/standardized_conv2d/kernel'.format(prefix)]))  # pylint: disable=line-too-long

KNOWN_MODELS = OrderedDict([
    ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])
