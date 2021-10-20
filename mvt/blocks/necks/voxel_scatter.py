import torch
import torch.nn as nn

from ..block_builder import NECKS


@NECKS.register_module()
class VoxelScatter(nn.Module):
    def __init__(self, grid_shape, num_input_features=32):
        """
        Converts learned features from dense tensor to sparse 3d grids.
        :param output_shape: ([int]: 5). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """
        super().__init__()
        self.name = 'VoxelsScatter'
        self.nz = grid_shape[0]
        self.ny = grid_shape[1]
        self.nx = grid_shape[2]
        self.nchannels = num_input_features

    def init_weights(self):
        """Initialize the weights of module."""
        pass

    def forward(self, voxel_features, coords, batch_size):
        """get the feature map to describe 3d grid of voxels
        :param voxel_features: [batch, max_voxels, num_input_features]
        :param coords: the coordinates of voxels
        :param batch_size: the size of batch
        :return: the feature map to describe 3d grid of voxels
        """
        # batch_canvas will be the final output.
        batch_canvas = torch.zeros(
            batch_size,
            self.nchannels,
            self.nx * self.ny * self.nz,
            dtype=voxel_features.dtype,
            device=voxel_features.device
        )

        for batch_itt in range(batch_size):
            # Only include non-empty voxels
            if len(coords.shape) == 2:
                this_coords = coords
                voxels = voxel_features
            else:
                this_coords = coords[batch_itt]
                if batch_size == 1:
                    voxels = voxel_features
                else:
                    voxels = voxel_features[batch_itt]

            indices = this_coords[..., 0] * self.nx * self.ny + \
                this_coords[..., 1] * self.nx + this_coords[..., 2]
            indices = indices.type(torch.long)

            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            batch_canvas[batch_itt, :, indices] = voxels

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        # batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(
            batch_size, self.nchannels, self.nz, self.ny, self.nx)

        return batch_canvas
