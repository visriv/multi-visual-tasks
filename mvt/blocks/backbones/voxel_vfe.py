import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

from mvt.utils.init_util import constant_init, kaiming_init
from mvt.utils.checkpoint_util import load_checkpoint
from ..block_builder import BACKBONES


class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels, last_layer=False):
        """ Voxel Feature Etractor Layer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """
        super().__init__()
        self.name = 'VFELayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = nn.BatchNorm1d(self.units, eps=1e-3, momentum=0.01)

    def forward(self, inputs):
        x = inputs.view(-1, inputs.shape[-2], inputs.shape[-1])
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        if self.last_vfe:
            x_max = torch.max(x, dim=-2, keepdim=True)[0]
            tmp_shape = list(inputs.shape[:-2]) + [-1, self.units]
            return x_max.view(tmp_shape)
        else:
            x_max = torch.max(x, dim=-2, keepdim=True)[0]
            x_repeat = x_max.repeat(1, inputs.shape[-2], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=-1)
            tmp_shape = list(inputs.shape[:-2]) + [-1, 2*self.units]
            return x_concatenated.view(tmp_shape)


class VFEATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, last_layer=False):
        """ Voxel Feature Etractor Layer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """
        super().__init__()
        self.name = 'VFEATLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        else:
            self.linear_at = nn.Linear(out_channels, 1, bias=True)

        self.units = out_channels

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = nn.BatchNorm1d(self.units, eps=1e-3, momentum=0.01)

    def forward(self, inputs):
        x = inputs.view(-1, inputs.shape[-2], inputs.shape[-1])
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        if self.last_vfe:
            x_a = self.linear_at(x)
            x_a = F.softmax(x_a, dim=-2)
            x = torch.sum(torch.mul(x, x_a), dim=-2)
            # x_max = torch.max(x, dim=-2, keepdim=True)[0]
            tmp_shape = list(inputs.shape[:-2]) + [-1, self.units]
            return x.view(tmp_shape)
        else:
            x_max = torch.max(x, dim=-2, keepdim=True)[0]
            x_repeat = x_max.repeat(1, inputs.shape[-2], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=-1)
            tmp_shape = list(inputs.shape[:-2]) + [-1, 2*self.units]
            return x_concatenated.view(tmp_shape)


@BACKBONES.register_module()
class VoxelMeanVFE(nn.Module):
    def __init__(self, num_input_features=4, num_filters=(64,),
                 voxel_size=[0.2, 0.2, 4], pc_range=[0., -40., -3.0, 70.4, 40., 1.0]):
        """Voxel Feature Net.
        The network prepares the voxel features and performs forward pass through VFELayers.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        """
        super().__init__()
        self.name = 'VoxelMeanVFE'

        # Create voxelFeatureNet layers
        num_output_filters = [num_input_features+6] + list(num_filters)
        vfe_layers = []
        for i in range(len(num_output_filters) - 1):
            in_filters = num_output_filters[i]
            out_filters = num_output_filters[i + 1]
            if i < len(num_output_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            vfe_layers.append(VFEATLayer(in_filters, out_filters, last_layer=last_layer))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.z_offset = self.vz / 2 + pc_range[2]

    def init_weights(self, pretrained=True):
        """Initialize the weights of module."""
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm1d):
                    constant_init(m, 1)

    def forward(self, features, num_points, coords):
        """get the features for voxels using geometric calculating and 
            voxel feature extractor
        :param features: input voxel features
        :param num_points: input number of points in voxels
        :return: voxel features
        """
        # Find distance of x, y, and z from cluster center
        dtype = features.dtype
        points_mean = features[..., :3].sum(dim=-2, keepdim=True)
        tmp_num = torch.where(
            num_points == 0, torch.full_like(num_points, 1), num_points)
        points_mean = points_mean/tmp_num.type_as(
            features).unsqueeze(-1).unsqueeze(-1)
        f_cluster = features[..., :3] - points_mean

        v_ci = coords.view(-1, coords.shape[-2], 3)
        f_center = torch.zeros_like(features[..., :3])
        f_center[..., 0] = features[..., 0] - (
            v_ci[:, :, 2].to(dtype).unsqueeze(-1) * self.vx + self.x_offset)
        f_center[..., 1] = features[..., 1] - (
            v_ci[:, :, 1].to(dtype).unsqueeze(-1) * self.vy + self.y_offset)
        f_center[..., 2] = features[..., 2] - (
            v_ci[:, :, 0].to(dtype).unsqueeze(-1) * self.vz + self.z_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        features = torch.cat(features_ls, dim=-1)

        # Forward pass through PFNLayers
        for vfe in self.vfe_layers:
            features = vfe(features)

        return features.squeeze()


@BACKBONES.register_module()
class VoxelCenterVFE(torch.nn.Module):
    def __init__(self, in_fdim, out_fdim, voxel_size, pc_range):
        super().__init__()
        self.name = 'VoxelCenterVFE'
        self.out_fdim = out_fdim
        self.linear = nn.Linear(in_fdim + 6, out_fdim, bias=False)
        self.batchnorm = nn.BatchNorm1d(out_fdim, eps=1e-3, momentum=0.01)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.z_offset = self.vz / 2 + pc_range[2]

    def forward(self, features, coors):
        """
        This module performs a Shared-based MLP. (both normal and strided version)
        :param points: float32[n_points, dim] - input points (center of neighborhoods)
        """
        dtype = features.dtype
        pos_in = features[..., :3]
        feature_in = features[..., 3:]
        x_p = pos_in.view(-1, pos_in.shape[-2], pos_in.shape[-1])
        x_f = feature_in.view(-1, feature_in.shape[-2], feature_in.shape[-1])
        v_ci = coors.view(-1, coors.shape[-2], 3)
        v_c = torch.zeros_like(v_ci, dtype=dtype)
        v_c[:, :, 0] = v_ci[:, :, 2].to(dtype) * self.vx + self.x_offset
        v_c[:, :, 1] = v_ci[:, :, 1].to(dtype) * self.vy + self.y_offset
        v_c[:, :, 2] = v_ci[:, :, 0].to(dtype) * self.vz + self.z_offset
        c_p = torch.zeros_like(x_p, dtype=dtype)
        c_p[:, :, 0] = x_p[:, :, 0] - v_c[:, :, 0]
        c_p[:, :, 1] = x_p[:, :, 1] - v_c[:, :, 1]
        c_p[:, :, 2] = x_p[:, :, 2] - v_c[:, :, 2]
        # c_d = torch.sum(torch.mul(c_p, c_p), dim=-1, keepdim=True)
        x = torch.cat([x_p, x_f, v_c, c_p], dim=-1)

        # Apply network [n_points, out_fdim]
        x = self.linear(x)
        x = self.batchnorm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        x_shape = list(pos_in.shape[:2]) + [self.out_fdim]

        return x.view(x_shape).squeeze()
