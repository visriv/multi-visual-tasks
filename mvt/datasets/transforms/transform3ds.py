import time
import numpy as np

from mvt.utils.bbox3d_util import (
    points_count_rbbox,
    limit_period
)
from mvt.utils.voxel_util import points_to_voxel
from mvt.datasets.data_converter.data_converter import (
    dc_dict_select,
    noise_per_object_v3_,
    random_flip,
    global_rotation_v2,
    global_scaling_v2,
    filter_gt_box_outside_range_by_center
)
from ..data_wrapper import PIPELINES
from .formating import to_tensor


@PIPELINES.register_module()
class PointCloudAlbu:
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(
        self,
        max_points_per_voxel,
        max_voxels=20000,
        create_gt=True,
        shuffle_points=False,
        gt_rotation_noise=(-np.pi/3, np.pi/3),
        gt_loc_noise_std=(1.0, 1.0, 1.0),
        global_rotation_noise=(-np.pi/4, np.pi/4),
        global_scaling_noise=(0.95, 1.05),
        min_points_in_gt=-1,
        random_flip_x=True,
        random_flip_y=True
    ):
        """convert point cloud to voxels, create targets if groundtruth exists.

        Args:
            max_points_per_voxel (int): the max number of points that used to 
                describe a voxel
            max_voxels (int, optional): max number of voxels to describe 
                a frame of point cloud. Defaults to 20000.
            create_gt (bool, optional): whether create new gt. Defaults to True.
            shuffle_points (bool, optional): whether shuffling points. 
                Defaults to False.
            gt_rotation_noise (tuple, optional): the range of ground truth 
                rotation noise. Defaults to (-np.pi/3, np.pi/3).
            gt_loc_noise_std (tuple, optional): the std error of ground truth 
                location noise. Defaults to (1.0, 1.0, 1.0).
            global_rotation_noise (tuple, optional): the range of global 
                rotation
                noise. Defaults to (-np.pi/4, np.pi/4).
            global_scaling_noise (tuple, optional): the range of global scaling 
                noise. Defaults to (0.95, 1.05).
            min_points_in_gt (int, optional): min number of points in each box. 
                Defaults to -1.
            random_flip_x (bool, optional): whether using random flip 
                around x axis. Defaults to True.
            random_flip_y (bool, optional): whether using random flip 
                around y axis. Defaults to True.
        """
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        self.create_gt = create_gt
        self.shuffle_points = shuffle_points
        self.gt_rotation_noise = gt_rotation_noise
        self.gt_loc_noise_std = gt_loc_noise_std
        self.global_rotation_noise = global_rotation_noise
        self.global_scaling_noise = global_scaling_noise
        self.min_points_in_gt = min_points_in_gt
        self.random_flip_x = random_flip_x
        self.random_flip_y = random_flip_y

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """
        class_names = results["target_assigner"].classes
        points = results["points"]
        metrics = {}
        if ("annotations" in results) and self.create_gt:
            anno_dict = results["annotations"]
            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": anno_dict["names"],
                "gt_importance": np.ones(
                    [anno_dict["boxes"].shape[0]], dtype=anno_dict["boxes"].dtype)
            }

            selected = np.array([i for i, x in enumerate(gt_dict["gt_names"])], dtype=np.int64)
            dc_dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                # points_count_rbbox takes 10ms with 10 sweeps nuscenes data
                point_counts = points_count_rbbox(points, gt_dict["gt_boxes"])
                mask = point_counts >= self.min_points_in_gt
                dc_dict_select(gt_dict, mask)
            gt_boxes_mask = np.array([n in class_names for n in gt_dict["gt_names"]], dtype=np.bool_)

            noise_per_object_v3_(
                gt_dict["gt_boxes"],
                points,
                gt_boxes_mask,
                rotation_perturb=self.gt_rotation_noise,
                center_noise_std=self.gt_loc_noise_std,
                num_try=100)

            dc_dict_select(gt_dict, gt_boxes_mask)
            gt_classes = np.array([class_names.index(n) + 1 for n in gt_dict["gt_names"]], dtype=np.int32)
            gt_dict["gt_classes"] = gt_classes
            gt_dict["gt_boxes"], points = random_flip(
                gt_dict["gt_boxes"],
                points,
                0.5,
                self.random_flip_x,
                self.random_flip_y)
            gt_dict["gt_boxes"], points = global_rotation_v2(
                gt_dict["gt_boxes"], 
                points, 
                *self.global_rotation_noise)
            gt_dict["gt_boxes"], points = global_scaling_v2(
                gt_dict["gt_boxes"], 
                points, 
                *self.global_scaling_noise)

            bv_range = results["point_cloud_range"][[0, 1, 3, 4]]
            # bv_range[2] = 0
            # bv_range[3] = 0
            mask = filter_gt_box_outside_range_by_center(
                gt_dict["gt_boxes"],
                bv_range
            )
            dc_dict_select(gt_dict, mask)

            # limit rad to [-pi, pi]
            gt_dict["gt_boxes"][:, 6] = limit_period(
                gt_dict["gt_boxes"][:, 6],
                offset=0.5,
                period=2 * np.pi
            )

        if self.shuffle_points:
            # shuffle is a little slow.
            np.random.shuffle(points)

        t1 = time.time()

        voxels, coordinates, num_points = points_to_voxel(
            points,
            voxel_size=results["voxel_size"],
            coors_range=results["point_cloud_range"],
            max_points=self.max_points_per_voxel,
            max_voxels=self.max_voxels
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        metrics["voxel_gene_time"] = time.time() - t1
        example = {
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coordinates,
            "num_voxels": num_voxels,
            "metrics": metrics,
        }

        anchors = results["anchor_cache"]["anchors"]
        matched_thresholds = results["anchor_cache"]["matched_thresholds"]
        unmatched_thresholds = results["anchor_cache"]["unmatched_thresholds"]

        example["anchors"] = anchors

        if ("annotations" in results) and self.create_gt:
            example["gt_names"] = gt_dict["gt_names"]
            targets_dict = results["target_assigner"].assign_all(
                anchors,
                gt_dict["gt_boxes"],
                gt_classes=gt_dict["gt_classes"],
                matched_thresholds=matched_thresholds,
                unmatched_thresholds=unmatched_thresholds
            )

            example.update({
                'labels': targets_dict['labels'],
                'reg_targets': targets_dict['bbox_targets'],
            })

        return example


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(max_points_per_voxel={self.max_points_per_voxel}, "
        repr_str += f"max_voxels={self.max_voxels}, "
        repr_str += f"create_gt={self.create_gt}, "
        repr_str += f"shuffle_points={self.shuffle_points})"
        repr_str += f"gt_rotation_noise={self.gt_rotation_noise}, "
        repr_str += f"gt_loc_noise_std={self.gt_loc_noise_std}, "
        repr_str += f"global_rotation_noise={self.global_rotation_noise}, "
        repr_str += f"global_scaling_noise={self.global_scaling_noise})"
        repr_str += f"min_points_in_gt={self.min_points_in_gt}, "
        repr_str += f"random_flip_x={self.random_flip_x}, "
        repr_str += f"random_flip_y={self.random_flip_y})"
        return repr_str


@PIPELINES.register_module()
class D3DDefaultFormatBundle:
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields.
    These fields are formatted as follows.
    - voxels: (1)to tensor.
    - num_points: (1)to tensor.
    - coordinates: (1)to tensor.
    - num_voxels: (1)to tensor.
    - metrics: (1)to tensor.
    - anchors: (1)to tensor.
    - labels: (1)to tensor.
    - reg_targets: (1)to tensor.
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """
        for key in [
            "voxels",
            "num_points",
            "coordinates",
            "num_voxels",
            "metrics",
            "anchors",
            "labels",
            "reg_targets"
        ]:
            if key not in results:
                continue
            results[key] = to_tensor(results[key])

        return results

    def __repr__(self):

        return self.__class__.__name__


@PIPELINES.register_module()
class D3DCollect:
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to collect keys in results. 
        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(keys={self.keys})"
        )
