import numpy as np
from mvt.utils.bbox3d_util import create_anchors_3d_range


# Anchor
class AnchorGenerator:
    """the basic class for anchor generator"""
    @property
    def size_name(self):
        raise NotImplementedError

    @property
    def num_anchors_per_localization(self):
        raise NotImplementedError

    def generate(self, feature_map_size):
        raise NotImplementedError

    @property
    def ndim(self):
        raise NotImplementedError


class AnchorGeneratorRange(AnchorGenerator):
    def __init__(
        self,
        anchor_ranges,
        sizes=[3.9, 1.6, 1.56],
        rotations=[0, np.pi / 2],
        size_name=None,
        match_threshold=-1,
        unmatch_threshold=-1,
        custom_values=(),
        dtype=np.float32
    ):
        """generate anchors by range
        :param anchor_ranges: range of point cloud for generate anchors
        :param sizes: the sizes of objects
        :param rotations: the list of rotations
        :param size_name: the name of size
        :param match_threshold: list of matching threshold
        :param unmatch_threshold: list of un-matching threshold
        :param custom_values: custom values for matching
        :param dtype: type of data
        """
        super().__init__()
        self._sizes = sizes
        self._anchor_ranges = anchor_ranges
        self._rotations = rotations
        self._dtype = dtype
        self._size_name = size_name
        self.match_threshold = match_threshold
        self.unmatch_threshold = unmatch_threshold
        self._custom_values = custom_values

    @property
    def size_name(self):
        return self._size_name

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def generate(self, feature_map_size):
        """
        generate anchors
        :param feature_map_size: the size of proposal feature map
        :return:
        """
        res = create_anchors_3d_range(
            feature_map_size,
            self._anchor_ranges,
            self._sizes,
            self._rotations,
            self._dtype)

        if len(self._custom_values) > 0:
            custom_ndim = len(self._custom_values)
            custom = np.zeros([*res.shape[:-1], custom_ndim])
            custom[:] = self._custom_values
            res = np.concatenate([res, custom], axis=-1)
        return res

    @property
    def ndim(self):
        return 7 + len(self._custom_values)

    @property
    def custom_ndim(self):
        return len(self._custom_values)
