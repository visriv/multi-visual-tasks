from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import roi_align


class RoIAlign(nn.Module):
    """RoI align pooling layer.

    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
        aligned (bool): if False, use the legacy implementation in
            Detection. If True, align the results more perfectly.

    Note:
        The implementation of RoIAlign when aligned=True is modified from
        https://github.com/facebookresearch/detectron2/
        The meaning of aligned=True:
        Given a continuous coordinate c, its two neighboring pixel
        indices (in our pixel model) are computed by floor(c - 0.5) and
        ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
        indices [0] and [1] (which are sampled from the underlying signal
        at continuous coordinates 0.5 and 1.5). But the original roi_align
        (aligned=False) does not subtract the 0.5 when computing
        neighboring pixel indices and therefore it uses pixels with a
        slightly incorrect alignment (relative to our pixel model) when
        performing bilinear interpolation.
        With `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5
        prior to calling roi_align. This produces the correct neighbors;
        The difference does not make a difference to the model's
        performance if ROIAlign is used together with conv layers.
    """

    def __init__(
        self,
        output_size,
        spatial_scale=1.0,
        sampling_ratio=0,
        pool_mode="avg",
        aligned=True,
    ):
        super(RoIAlign, self).__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N.\
                The other 4 columns are xyxy.
        """
        if "aligned" in roi_align.__code__.co_varnames:
            return roi_align(
                input,
                rois,
                self.output_size,
                self.spatial_scale,
                self.sampling_ratio,
                self.aligned,
            )
        else:
            if self.aligned:
                rois -= rois.new_tensor([0.0] + [0.5 / self.spatial_scale] * 4)
            return roi_align(
                input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
            )

    def __repr__(self):
        s = self.__class__.__name__
        s += f"(output_size={self.output_size}, "
        s += f"spatial_scale={self.spatial_scale}, "
        s += f"sampling_ratio={self.sampling_ratio}, "
        s += f"pool_mode={self.pool_mode}, "
        s += f"aligned={self.aligned}, "
        return s
