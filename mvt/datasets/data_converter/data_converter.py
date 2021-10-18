import numpy as np
import time
import numba
from mvt.utils.bbox3d_util import (
    points_in_convex_polygon_3d_jit, 
    points_in_convex_polygon_jit,
    box2d_to_corner_jit,
    minmax_to_corner_2d,
    center_to_corner_box3d,
    corner_to_surfaces_3d_jit,
    rotation_points_single_angle,
    corner_to_standup_nd_jit,
    points_count_rbbox,
    limit_period
)

from mvt.utils.voxel_util import (
    points_to_voxel,
    dyn_points_to_voxel
)


def read_dataset_file(path):
    """
    read the dataset information file by lines
    :param path: the path of the dataset info file
    :return: the list of data information
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def filter_gt_box_outside_range_by_center(gt_boxes, limit_range):
    """
    remove gtbox outside training range. this function should be applied after other prep functions
    :param gt_boxes: ground truth boxes [N_box, 7]
    :param limit_range: the limited range of object centers [6,]
    :return: [N_box,] bool array for determining whether a box in or out the limit_range
    """
    gt_box_centers = gt_boxes[:, :2]
    bounding_box = minmax_to_corner_2d(np.asarray(limit_range)[np.newaxis, ...])
    ret = points_in_convex_polygon_jit(gt_box_centers, bounding_box)
    return ret.reshape(-1)


@numba.njit
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    """
    get the 3d rotational matrix from the given angle
    :param rot_mat_T: the rotational matrix
    :param angle: the given angle
    :param axis: the rotational axis
    :return:
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = rot_sin
        rot_mat_T[2, 0] = -rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = rot_sin
        rot_mat_T[1, 0] = -rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = rot_sin
        rot_mat_T[2, 1] = -rot_sin
        rot_mat_T[2, 2] = rot_cos


@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_T):
    """
    get the 2d rotational matrix from angle, and then calculate the corners after rotating.
    :param corners: the 2d box corners
    :param angle: the given rotational angle
    :param rot_mat_T: the 2d rotational matrix
    :return:
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = rot_sin
    rot_mat_T[1, 0] = -rot_sin
    rot_mat_T[1, 1] = rot_cos
    for i in range(len(corners)):
        for j in range(2):
            corners[i, j] = corners[i, 0]*rot_mat_T[0, j] + corners[i, 1]*rot_mat_T[1, j]
    # corners[:] = corners @ rot_mat_T


@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    """
    noising each box with locational and rotational noises
    :param boxes: [N, 5], locations and rotations for boxes
    :param valid_mask: [N], whether the box is valid
    :param loc_noises: [N, M, 3], M noise types for location
    :param rot_noises: [N, M], M noise types for rotation
    :return: [N], noising type
    """
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask


@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises, rot_noises, global_rot_noises):
    """
    noising each box with locational and rotational noises
    :param boxes: [N, 5], locations and rotations for boxes
    :param valid_mask: [N], whether the box is valid
    :param loc_noises: [N, M, 3], M noise types for location
    :param rot_noises: [N, M], M noise types for rotation
    :param global_rot_noises: [N, M], M noise types for the global rotation
    :return: [N], noising type
    """
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2,), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                current_radius = np.sqrt(boxes[i, 0] ** 2 + boxes[i, 1] ** 2)
                current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.sin(dst_grot)
                dst_pos[1] = current_radius * np.cos(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += (dst_grot - current_grot)

                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_T[0, 0] = rot_cos
                rot_mat_T[0, 1] = rot_sin
                rot_mat_T[1, 0] = -rot_sin
                rot_mat_T[1, 1] = rot_cos
                for m in range(4):
                    for n in range(2):
                        current_corners[m, n] = current_box[0, 2]*corners_norm[m, 0]*rot_mat_T[0, n] + \
                                                current_box[0, 3] * corners_norm[m, 1] * rot_mat_T[1, n] + current_box[0, n]

                # current_corners[:] = current_box[0, 2:4] * corners_norm @ rot_mat_T + current_box[0, :2]

                current_corners -= current_box[0, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += (dst_pos - boxes[i, :2])
                    rot_noises[i, j] += (dst_grot - current_grot)
                    break

    return success_mask


@numba.njit
def points_transform_(points, centers, point_masks, loc_transform, rot_transform, valid_mask):
    """
    transform location and rotation with points
    :param points: [N_p, 3+]
    :param centers: [N_b, 3]
    :param point_masks: [N_p, N_b]
    :param loc_transform: [N_b, 3]
    :param rot_transform: [N_b, ]
    :param valid_mask: [N_b] indicate the valid box
    :return:
    """
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i, 0] = points[i, 0]*rot_mat_T[j, 0, 0] + points[i, 1]*rot_mat_T[j, 1, 0] + points[i, 2]*rot_mat_T[j, 2, 0]
                    points[i, 1] = points[i, 0]*rot_mat_T[j, 0, 1] + points[i, 1]*rot_mat_T[j, 1, 1] + points[i, 2]*rot_mat_T[j, 2, 1]
                    points[i, 2] = points[i, 0]*rot_mat_T[j, 0, 2] + points[i, 1]*rot_mat_T[j, 1, 2] + points[i, 2]*rot_mat_T[j, 2, 2]
                    # points[i:i + 1, :3] = points[i:i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform


@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    """
    3d transformation for boxes
    :param boxes: [N, 7]
    :param loc_transform: [N, 3]
    :param rot_transform: [N,]
    :param valid_mask: [N,] indicate the valid box
    :return:
    """
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


def _select_transform(transform, indices):
    """
    select the specific transformations
    :param transform: [N, M] M types transformations for N objects
    :param indices: [N,] select the specific transformation for an object
    :return: selected transformations
    """
    result = np.zeros((transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result


def noise_per_object_v3_(gt_boxes, points=None, valid_mask=None,
                         rotation_perturb=np.pi/4, center_noise_std=1.0, num_try=5):
    """
    random rotate or remove each groundtruth independently.
    :param gt_boxes: groud truth boxes
    :param points: needed when transforming points
    :param valid_mask: whether the box is valid
    :param rotation_perturb: rotation range for uniform noise
    :param center_noise_std: the std error for Gaussian noise
    :param num_try: int, try number for random
    :return:
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]

    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [center_noise_std, center_noise_std, center_noise_std]
    if all([c == 0 for c in center_noise_std]) and all([c == 0 for c in rotation_perturb]):
        return
    if valid_mask is None:
        valid_mask = np.ones((num_boxes,), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(scale=center_noise_std, size=[num_boxes, num_try, 3])

    rot_noises = np.random.uniform(rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])

    origin = [0.5, 0.5, 0.5]
    gt_box_corners = center_to_corner_box3d(
        gt_boxes[:, :3],
        gt_boxes[:, 3:6],
        gt_boxes[:, 6],
        origin=origin,
        axis=2)

    selected_noise = noise_per_box(gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises, rot_noises)

    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    surfaces = corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(points, gt_boxes[:, :3], point_masks, loc_transforms, rot_transforms, valid_mask)

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)


def random_flip(gt_boxes, points, probability=0.5, random_flip_x=True, random_flip_y=True):
    """
    random flip for boxes and points
    :param gt_boxes: the ground truth boxes
    :param points: points
    :param probability: the probability of using the flip
    :param random_flip_x: whether flip around x axis
    :param random_flip_y: whether flip around y axis
    :return: the flipped boxes and points
    """
    flip_x = np.random.choice([False, True], replace=False, p=[1 - probability, probability])
    flip_y = np.random.choice([False, True], replace=False, p=[1 - probability, probability])
    if flip_y and random_flip_y:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi
        if gt_boxes.shape[1] == 9:
            gt_boxes[:, 8] = -gt_boxes[:, 8]
        points[:, 1] = -points[:, 1]
    if flip_x and random_flip_x:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        if gt_boxes.shape[1] == 9:
            gt_boxes[:, 7] = -gt_boxes[:, 7]
        points[:, 0] = -points[:, 0]

    return gt_boxes, points


def global_scaling_v2(gt_boxes, points, min_scale=0.95, max_scale=1.05):
    """
    scaling with boxes and points
    :param gt_boxes: boxes
    :param points: points
    :param min_scale: the lower limit of scale range
    :param max_scale: the upper limit of scale range
    :return: scaled boxes and points
    """
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    if gt_boxes.shape[1] == 9:
        gt_boxes[:, 7:] *= noise_scale
    return gt_boxes, points


def global_rotation_v2(gt_boxes, points, min_rad=-np.pi / 4, max_rad=np.pi / 4):
    """
    global ratation for boxes and points
    :param gt_boxes: boxes
    :param points: points
    :param min_rad: the lower limit of rotation range
    :param max_rad: the upper limit of rotation range
    :return: rotated boxes and points
    """
    noise_rotation = np.random.uniform(min_rad, max_rad)
    points[:, :3] = rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2)
    gt_boxes[:, :3] = rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2)
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] == 9:
        # rotate velo vector
        rot_cos = np.cos(noise_rotation)
        rot_sin = np.sin(noise_rotation)
        rot_mat_T = np.array([[rot_cos, rot_sin], [-rot_sin, rot_cos]], dtype=points.dtype)

        # gt_boxes[:, 7:9] = gt_boxes[:, 7:9] @ rot_mat_T
        gt_boxes[:, 7] = gt_boxes[:, 7] * rot_mat_T[0, 0] + gt_boxes[:, 8] * rot_mat_T[1, 0]
        gt_boxes[:, 8] = gt_boxes[:, 7] * rot_mat_T[0, 1] + gt_boxes[:, 8] * rot_mat_T[1, 1]

    return gt_boxes, points


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """
    test box collision
    :param boxes: [N, 7] boxes
    :param qboxes: [K, 7] query boxes
    :param clockwise: whether being along clockwise
    :return: [N, K]
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]), axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    boxes_standup = corner_to_standup_nd_jit(boxes)
    qboxes_standup = corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * (boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * (qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.

    return ret


def _dict_select(dict_, inds):
    """
    select key-values in dictionary
    :param dict_: dictionary
    :param inds: selected indices
    :return:
    """
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def prep_pointcloud(input_dict,
                    point_cloud_range,
                    voxel_size,
                    max_points_per_voxel,
                    feature_map_size,
                    target_assigner,
                    create_gt=True,
                    anchor_cache=None,
                    max_voxels=20000,
                    shuffle_points=False,
                    gt_rotation_noise=(-np.pi/3, np.pi/3),
                    gt_loc_noise_std=(1.0, 1.0, 1.0),
                    global_rotation_noise=(-np.pi/4, np.pi/4),
                    global_scaling_noise=(0.95, 1.05),
                    min_points_in_gt=-1,
                    random_flip_x=True,
                    random_flip_y=True):
    """
    convert point cloud to voxels, create targets if ground truths exists.
    :param input_dict: format: dataset.get_sensor_data format
    :param point_cloud_range: range of point cloud defined in config
    :param voxel_size: size of voxel
    :param max_points_per_voxel: the max number of points that used to describe a voxel
    :param feature_map_size: the size of feature map which used to infer objects
    :param target_assigner: a class operator used to assign ground truth boxes to anchors
    :param max_voxels: max number of voxels to describe a frame of point cloud
    :param training: whether training or not
    :param shuffle_points: whether shuffling points
    :param gt_rotation_noise: the range of ground truth rotation noise
    :param gt_loc_noise_std: the std error of ground truth location noise
    :param global_rotation_noise: the range of global rotation noise
    :param global_scaling_noise: the range of global scaling noise
    :param anchor_cache: if not None, using anchors in cache
    :param min_points_in_gt: min number of points in each box
    :param random_flip_x: whether using random flip around x axis
    :param random_flip_y: whether using random flip around y axis
    :return: the input for network
    """
    class_names = target_assigner.classes
    points = input_dict["points"]
    metrics = {}
    if create_gt:
        anno_dict = input_dict["annotations"]
        gt_dict = {
            "gt_boxes": anno_dict["boxes"],
            "gt_names": anno_dict["names"],
            "gt_importance": np.ones([anno_dict["boxes"].shape[0]], dtype=anno_dict["boxes"].dtype),
        }

        selected = np.array([i for i, x in enumerate(gt_dict["gt_names"])], dtype=np.int64)
        _dict_select(gt_dict, selected)

        if min_points_in_gt > 0:
            # points_count_rbbox takes 10ms with 10 sweeps nuscenes data
            point_counts = points_count_rbbox(points, gt_dict["gt_boxes"])
            mask = point_counts >= min_points_in_gt
            _dict_select(gt_dict, mask)
        gt_boxes_mask = np.array([n in class_names for n in gt_dict["gt_names"]], dtype=np.bool_)

        noise_per_object_v3_(
            gt_dict["gt_boxes"],
            points,
            gt_boxes_mask,
            rotation_perturb=gt_rotation_noise,
            center_noise_std=gt_loc_noise_std,
            num_try=100)

        _dict_select(gt_dict, gt_boxes_mask)
        gt_classes = np.array([class_names.index(n) + 1 for n in gt_dict["gt_names"]], dtype=np.int32)
        gt_dict["gt_classes"] = gt_classes
        gt_dict["gt_boxes"], points = random_flip(gt_dict["gt_boxes"], points, 0.5, random_flip_x, random_flip_y)
        gt_dict["gt_boxes"], points = global_rotation_v2(gt_dict["gt_boxes"], points, *global_rotation_noise)
        gt_dict["gt_boxes"], points = global_scaling_v2(gt_dict["gt_boxes"], points, *global_scaling_noise)

        bv_range = point_cloud_range[[0, 1, 3, 4]]
        # bv_range[2] = 0
        # bv_range[3] = 0
        mask = filter_gt_box_outside_range_by_center(gt_dict["gt_boxes"], bv_range)
        _dict_select(gt_dict, mask)

        # limit rad to [-pi, pi]
        gt_dict["gt_boxes"][:, 6] = limit_period(gt_dict["gt_boxes"][:, 6], offset=0.5, period=2 * np.pi)

    if shuffle_points:
        # shuffle is a little slow.
        np.random.shuffle(points)

    t1 = time.time()

    voxels, coordinates, num_points = points_to_voxel(
        points,
        voxel_size=voxel_size,
        coors_range=point_cloud_range,
        max_points=max_points_per_voxel,
        max_voxels=max_voxels)
    num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

    metrics["voxel_gene_time"] = time.time() - t1
    example = {
        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": num_voxels,
        "metrics": metrics,
    }

    if anchor_cache is not None:
        anchors = anchor_cache["anchors"]
        matched_thresholds = anchor_cache["matched_thresholds"]
        unmatched_thresholds = anchor_cache["unmatched_thresholds"]
    else:
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, target_assigner.box_ndim])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]

    example["anchors"] = anchors

    if create_gt:
        example["gt_names"] = gt_dict["gt_names"]
        targets_dict = target_assigner.assign_all(
            anchors,
            gt_dict["gt_boxes"],
            gt_classes=gt_dict["gt_classes"],
            matched_thresholds=matched_thresholds,
            unmatched_thresholds=unmatched_thresholds)

        example.update({
            'labels': targets_dict['labels'],
            'reg_targets': targets_dict['bbox_targets'],
        })

    return example


def prep_dyn_pointcloud(input_dict,
                        point_cloud_range,
                        voxel_size,
                        feature_map_size,
                        target_assigner,
                        create_gt=True,
                        anchor_cache=None,
                        shuffle_points=False,
                        gt_rotation_noise=(-np.pi/3, np.pi/3),
                        gt_loc_noise_std=(1.0, 1.0, 1.0),
                        global_rotation_noise=(-np.pi/4, np.pi/4),
                        global_scaling_noise=(0.95, 1.05),
                        min_points_in_gt=-1,
                        random_flip_x=True,
                        random_flip_y=True):
    """
    convert point cloud to voxels, create targets if ground truths exists.
    :param input_dict: format: dataset.get_sensor_data format
    :param point_cloud_range: range of point cloud defined in config
    :param voxel_size: size of voxel
    :param max_points_per_voxel: the max number of points that used to describe a voxel
    :param feature_map_size: the size of feature map which used to infer objects
    :param target_assigner: a class operator used to assign ground truth boxes to anchors
    :param max_voxels: max number of voxels to describe a frame of point cloud
    :param training: whether training or not
    :param shuffle_points: whether shuffling points
    :param gt_rotation_noise: the range of ground truth rotation noise
    :param gt_loc_noise_std: the std error of ground truth location noise
    :param global_rotation_noise: the range of global rotation noise
    :param global_scaling_noise: the range of global scaling noise
    :param anchor_cache: if not None, using anchors in cache
    :param min_points_in_gt: min number of points in each box
    :param random_flip_x: whether using random flip around x axis
    :param random_flip_y: whether using random flip around y axis
    :return: the input for network
    """
    class_names = target_assigner.classes
    points = input_dict["points"]
    metrics = {}
    if create_gt:
        anno_dict = input_dict["annotations"]
        gt_dict = {
            "gt_boxes": anno_dict["boxes"],
            "gt_names": anno_dict["names"],
            "gt_importance": np.ones([anno_dict["boxes"].shape[0]], dtype=anno_dict["boxes"].dtype),
        }

        selected = np.array([i for i, x in enumerate(gt_dict["gt_names"])], dtype=np.int64)
        _dict_select(gt_dict, selected)

        if min_points_in_gt > 0:
            # points_count_rbbox takes 10ms with 10 sweeps nuscenes data
            point_counts = points_count_rbbox(points, gt_dict["gt_boxes"])
            mask = point_counts >= min_points_in_gt
            _dict_select(gt_dict, mask)
        gt_boxes_mask = np.array([n in class_names for n in gt_dict["gt_names"]], dtype=np.bool_)

        noise_per_object_v3_(
            gt_dict["gt_boxes"],
            points,
            gt_boxes_mask,
            rotation_perturb=gt_rotation_noise,
            center_noise_std=gt_loc_noise_std,
            num_try=100)

        _dict_select(gt_dict, gt_boxes_mask)
        gt_classes = np.array([class_names.index(n) + 1 for n in gt_dict["gt_names"]], dtype=np.int32)
        gt_dict["gt_classes"] = gt_classes
        gt_dict["gt_boxes"], points = random_flip(gt_dict["gt_boxes"], points, 0.5, random_flip_x, random_flip_y)
        gt_dict["gt_boxes"], points = global_rotation_v2(gt_dict["gt_boxes"], points, *global_rotation_noise)
        gt_dict["gt_boxes"], points = global_scaling_v2(gt_dict["gt_boxes"], points, *global_scaling_noise)

        bv_range = point_cloud_range[[0, 1, 3, 4]]
        # bv_range[2] = 0
        # bv_range[3] = 0
        mask = filter_gt_box_outside_range_by_center(gt_dict["gt_boxes"], bv_range)
        _dict_select(gt_dict, mask)

        # limit rad to [-pi, pi]
        gt_dict["gt_boxes"][:, 6] = limit_period(gt_dict["gt_boxes"][:, 6], offset=0.5, period=2 * np.pi)

    if shuffle_points:
        # shuffle is a little slow.
        np.random.shuffle(points)

    t1 = time.time()

    valid_points, coor_voxelidx = dyn_points_to_voxel(
        points,
        voxel_size=voxel_size,
        coors_range=point_cloud_range)

    metrics["voxel_gene_time"] = time.time() - t1
    example = {
        'valid_points': valid_points,
        'coor_voxelidx': coor_voxelidx,
        "metrics": metrics,
    }

    if anchor_cache is not None:
        anchors = anchor_cache["anchors"]
        matched_thresholds = anchor_cache["matched_thresholds"]
        unmatched_thresholds = anchor_cache["unmatched_thresholds"]
    else:
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, target_assigner.box_ndim])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]

    example["anchors"] = anchors

    if create_gt:
        example["gt_names"] = gt_dict["gt_names"]
        targets_dict = target_assigner.assign_all(
            anchors,
            gt_dict["gt_boxes"],
            gt_classes=gt_dict["gt_classes"],
            matched_thresholds=matched_thresholds,
            unmatched_thresholds=unmatched_thresholds)

        example.update({
            'labels': targets_dict['labels'],
            'reg_targets': targets_dict['bbox_targets'],
        })

    return example
