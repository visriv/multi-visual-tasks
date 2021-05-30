import torch.nn as nn
from torch.nn.modules.utils import _pair
from torchvision.ops import roi_pool


class RoIPool(nn.Module):

    def __init__(self, output_size, spatial_scale=1.0):
        super(RoIPool, self).__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)

    def forward(self, input, rois):
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale})'
        return s
