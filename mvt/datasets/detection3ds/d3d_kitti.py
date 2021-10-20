from pathlib import Path
import numpy as np

from mvt.datasets.data_wrapper import DATASETS
from .d3d_base import D3dBaseDataset


@DATASETS.register_module()
class KittiDataset(D3dBaseDataset):
    """the dataset class derived from an abstract class 'Dataset'."""

    class_names = ('car', 'pedestrian', 'cyclist')

    def __init__(self, data_cfg, pipeline_cfg, root_path, net=None, sel_index=0):
        """
        initialization
        :param root_path: the root path for dataset
        :param info_path: the path of dataset information files
        :param class_names: class names in the dataset
        :param prep_func: function for pre-processing the data
        """
        super(KittiDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, net, sel_index)
        # the number of input point cloud features, usually are x, y, z, intensity.
        self.num_pf = 4
        out_size_factor = data_cfg.RPN_STRIDES
        voxel_size = data_cfg.VOXEL_SIZE
        point_cloud_range = data_cfg.POINT_CLOUD_RANGE
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        feature_map_size = grid_size[:2] // out_size_factor
        self.feature_map_size = [*feature_map_size, 1][::-1]

        self.target_assigner = net.target_assigner
        ret = self.target_assigner.generate_anchors(self.feature_map_size)

        anchors = ret["anchors"].reshape(-1, self.target_assigner.box_ndim)
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        self.anchor_cache = {
            "anchors": anchors,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
        }
        

    def get_sensor_data(self, idx):
        """
        Dataset must provide a unified function to get data.
        :param idx: id or dict for querying data from the dataset
        :return: sensor data and its annotation
        """
        info = self.data_infos[idx]
        results = {
            "data_idx": info["data_idx"],
            "type": "lidar",
            "points": None,
            "point_cloud_range": self.point_cloud_range,
            "voxel_size": self.voxel_size,
            "feature_map_size": self.feature_map_size,
            "target_assigner": self.target_assigner,
            "anchor_cache": self.anchor_cache
        }

        velo_path = Path(info['pc_path'])
        if not velo_path.is_absolute():
            velo_path = Path(self.data_root) / info['pc_path']

        point_cloud = np.fromfile(
            str(velo_path), dtype=np.float32, count=-1).reshape(
                [-1, self.num_pf])

        results["points"] = point_cloud.reshape([-1, 4])

        if 'annos' in info:
            annos = info['annos']
            # we need other objects to avoid collision when sample
            gt_names = annos["names"]
            locs = annos["center_xyz"]
            dims = annos["size_lwh"]
            rots = annos["rot_theta"]
            gt_boxes = np.concatenate(
                [locs, dims, rots[..., np.newaxis]], axis=1).astype(
                    np.float32)

            # only center format is allowed. so we need to convert
            results["annotations"] = {
                'names': gt_names,
                'boxes': gt_boxes,
            }

        return results
