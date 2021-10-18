import numpy as np
import numba


@numba.njit
def surface_equ_3d_jitv2(surfaces):
    """
    calculate the normal vectors of surfaces
    :param surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    :return: the normal vectors and direction factor
    """
    num_polygon = surfaces.shape[0]
    max_num_surfaces = surfaces.shape[1]
    normal_vec = np.zeros((num_polygon, max_num_surfaces, 3), dtype=surfaces.dtype)
    d = np.zeros((num_polygon, max_num_surfaces), dtype=surfaces.dtype)
    sv0 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    sv1 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    for i in range(num_polygon):
        for j in range(max_num_surfaces):
            sv0[0] = surfaces[i, j, 0, 0] - surfaces[i, j, 1, 0]
            sv0[1] = surfaces[i, j, 0, 1] - surfaces[i, j, 1, 1]
            sv0[2] = surfaces[i, j, 0, 2] - surfaces[i, j, 1, 2]
            sv1[0] = surfaces[i, j, 1, 0] - surfaces[i, j, 2, 0]
            sv1[1] = surfaces[i, j, 1, 1] - surfaces[i, j, 2, 1]
            sv1[2] = surfaces[i, j, 1, 2] - surfaces[i, j, 2, 2]
            normal_vec[i, j, 0] = (sv0[1] * sv1[2] - sv0[2] * sv1[1])
            normal_vec[i, j, 1] = (sv0[2] * sv1[0] - sv0[0] * sv1[2])
            normal_vec[i, j, 2] = (sv0[0] * sv1[1] - sv0[1] * sv1[0])

            d[i, j] = - surfaces[i, j, 0, 0] * normal_vec[i, j, 0] \
                      - surfaces[i, j, 0, 1] * normal_vec[i, j, 1] \
                      - surfaces[i, j, 0, 2] * normal_vec[i, j, 2]
    return normal_vec, d


def points_in_convex_polygon_jit(points, polygon, clockwise=True):
    """
    check points is in 2d convex polygons. True when point in polygon
    :param points: [num_points, 2] array.
    :param polygon: [num_polygon, num_points_of_polygon, 2] array.
    :param clockwise: bool. indicate polygon is clockwise.
    :return: [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_points_of_polygon = polygon.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon.shape[0]
    if clockwise:
        vec1 = polygon - polygon[:, [num_points_of_polygon - 1] + list(range(num_points_of_polygon - 1)), :]
    else:
        vec1 = polygon[:, [num_points_of_polygon - 1] + list(range(num_points_of_polygon - 1)), :] - polygon

    ret = np.zeros((num_points, num_polygons), dtype=np.bool_)

    for i in range(num_points):
        for j in range(num_polygons):
            success = True
            for k in range(num_points_of_polygon):
                cross = vec1[j, k, 1] * (polygon[j, k, 0] - points[i, 0])
                cross -= vec1[j, k, 0] * (polygon[j, k, 1] - points[i, 1])
                if cross >= 0:
                    success = False
                    break
            ret[i, j] = success
    return ret


@numba.njit
def _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d, num_surfaces=None):
    """
    check points is in 3d convex polygons.
    :param points: input points
    :param polygon_surfaces: [num_polygon, max_num_surfaces, max_num_points_of_surface, 3] array.
           all surfaces' normal vector must direct to internal. max_num_points_of_surface must at least 3.
    :param normal_vec: the normal vectors of surfaces
    :param d: direction factor
    :param num_surfaces: [num_polygon] array. indicate how many surfaces a polygon contain
    :return: [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] \
                       + points[i, 1] * normal_vec[j, k, 1] \
                       + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    """
    check points is in 3d convex polygons.
    :param points: [num_points, 3] array.
    :param polygon_surfaces: [num_polygon, max_num_surfaces, max_num_points_of_surface, 3] array.
           all surfaces' normal vector must direct to internal. max_num_points_of_surface must at least 3.
    :param num_surfaces: [num_polygon] array. indicate how many surfaces a polygon contain
    :return: [num_points, num_polygon] bool array.
    """
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d, num_surfaces)


@numba.njit
def _points_count_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d, num_surfaces=None):
    """
    count points in 3d convex polygons.
    :param points: [num_points, 3] array.
    :param polygon_surfaces: [num_polygon, max_num_surfaces, max_num_points_of_surface, 3] array.
           all surfaces' normal vector must direct to internal. max_num_points_of_surface must at least 3.
    :param normal_vec: the normal vectors of surfaces
    :param d: direction factor
    :param num_surfaces: [num_polygon] array. indicate how many surfaces a polygon contain
    :return: [num_polygon] array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.full((num_polygons,), num_points, dtype=np.int64)
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] \
                       + points[i, 1] * normal_vec[j, k, 1] \
                       + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[j] -= 1
                    break
    return ret


def points_count_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    """
    check points is in 3d convex polygons.
    :param points: [num_points, 3] array.
    :param polygon_surfaces: [num_polygon, max_num_surfaces, max_num_points_of_surface, 3] array.
            all surfaces' normal vector must direct to internal. max_num_points_of_surface must at least 3.
    :param num_surfaces: [num_polygon] array. indicate how many surfaces a polygon contain
    :return: [num_polygon] array.
    """
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])

    return _points_count_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d, num_surfaces)


@numba.njit
def _points_index_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d, num_surfaces=None):
    """
    count points in 3d convex polygons.
    :param points: [num_points, 3] array.
    :param polygon_surfaces: [num_polygon, max_num_surfaces, max_num_points_of_surface, 3] array.
           all surfaces' normal vector must direct to internal. max_num_points_of_surface must at least 3.
    :param normal_vec: the normal vectors of surfaces
    :param d: direction factor
    :param num_surfaces: [num_polygon] array. indicate how many surfaces a polygon contain
    :return: [num_points] array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.full((num_points,), 1, dtype=np.int64)
    for i in range(num_points):
        is_inbox = False
        for j in range(num_polygons):
            is_inthisbox = True
            for k in range(max_num_surfaces):
                sign = points[i, 0] * normal_vec[j, k, 0] \
                       + points[i, 1] * normal_vec[j, k, 1] \
                       + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    is_inthisbox = False
                    break
            if is_inthisbox:
                is_inbox = True
                break
        if is_inbox:
            ret[i] = 0
    return ret


def points_index_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    """
    check points is in 3d convex polygons.
    :param points: [num_points, 3] array.
    :param polygon_surfaces: [num_polygon, max_num_surfaces, max_num_points_of_surface, 3] array.
           all surfaces' normal vector must direct to internal. max_num_points_of_surface must at least 3.
    :param num_surfaces: [num_polygon] array. indicate how many surfaces a polygon contain
    :return: [num_polygon] array.
    """
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])

    return _points_index_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d, num_surfaces)


def corners_nd(dims, origin=0.5):
    """
    generate relative box corners based on length per dim and origin point.
    :param dims: float array, shape=[N, ndim], array of length per dim
    :param origin: list or array or float, origin point relate to smallest point.
    :return: float array, shape=[N, 2 ** ndim, ndim]: returned corners.
             point layout example: x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
             where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


def corner_to_surfaces_3d(corners):
    """
    convert 3d box corners from corner function above to surfaces that normal vectors all direct to internal.
    :param corners: float array, [N, 8, 3], 3d box corners.
    :return: surfaces, float array, [N, 6, 4, 3]
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces


@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """
    convert 3d box corners from corner function above to surfaces that normal vectors all direct to internal.
    :param corners: float array, [N, 8, 3], 3d box corners.
    :return: surfaces, float array, [N, 6, 4, 3]
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array([0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7]).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces


@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    """
    convert box corners to stand-up box corners with numba
    :param boxes_corner: the given box corners
    :return: stand-up box corners that are along axises
    """
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


def corner_to_standup_nd(boxes_corner):
    """
    convert box corners to stand-up box corners
    :param boxes_corner: the given box corners
    :return: stand-up box corners that are along axises
    """
    assert len(boxes_corner.shape) == 3
    standup_boxes = []
    standup_boxes.append(np.min(boxes_corner, axis=1))
    standup_boxes.append(np.max(boxes_corner, axis=1))
    return np.concatenate(standup_boxes, -1)


@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes):
    """
    convert 2d boxes to corners
    :param boxes: 2d boxes
    :return: 2d corners
    """
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(1, 4, 2)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = rot_sin
        rot_mat_T[1, 0] = -rot_sin
        rot_mat_T[1, 1] = rot_cos
        for j in range(4):
            for k in range(2):
                box_corners[i, j, k] = corners[i, j, 0]*rot_mat_T[0, k] + corners[i, j, 1]*rot_mat_T[1, k]
        # box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners


def rbbox2d_to_near_bbox(rbboxes):
    """
    convert rotated bbox to nearest 'standing' or 'lying' bbox.
    :param rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    :return: bboxes, [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


def minmax_to_corner_2d(minmax_box):
    """
    get the minmax boxes
    :param minmax_box: boxes with minmax values for each axis
    :return: box corners
    """
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box2d(center, dims, origin=0.0)


def center_to_minmax_2d_0_5(centers, dims):
    """
    get the minmax boxes with centers and object sizes
    :param centers: centers
    :param dims: object sizes
    :return: box corners
    """
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """
    convert locations, dimensions and angles to corners. format: center(xy), dims(xy), angles(clockwise when positive)
    :param centers: float array, shape=[N, 2], locations in kitti label file.
    :param dims: float array, shape=[N, 2], dimensions in kitti label file.
    :param angles: float array, shape=[N], rotation_y in kitti label file.
    :param origin: list or array or float, origin point relate to smallest point.
    :return: float array, shape=[N, 4, 2], returned corners.
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def center_to_minmax_2d(centers, dims, origin=0.5):
    """
    convert centers and dims to minmax corners
    :param centers: box centers
    :param dims: object sizes
    :param origin: origin point relate to smallest point. [0.5, 0.5, 0.5] in lidar.
    :return: float array, shape=[N, 4, 2], returned corners.
    """
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 0.5, 0.5), axis=2):
    """
    convert locations, dimensions and angles to corners
    :param centers: float array, shape=[N, 3], loc_xyz in ecarx label file.
    :param dims: float array, shape=[N, 3], dim_hwl in ecarx label file.
    :param angles: float array, shape=[N], rot_theta in ecarx label file.
    :param origin: list or array or float, origin point relate to smallest point. [0.5, 0.5, 0.5] in lidar.
    :param axis: int, rotation axis. 2 for lidar.
    :return: float array, shape=[N, 8, 3], returned corners.
    """
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def rotation_2d(points, angles):
    """
    rotation 2d points based on origin point clockwise when angle positive.
    :param points: float array, shape=[N, point_size, 2], points to be rotated.
    :param angles: float array, shape=[N], rotational angle.
    :return: float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, rot_sin], [-rot_sin, rot_cos]])
    corner = np.einsum('aij,jka->aik', points, rot_mat_T)
    return corner


def rotation_3d_in_axis(points, angles, axis=0):
    """
    rotation 3d points based on origin point clockwise when angle positive.
    :param points: float array, shape=[N, point_size, 3], points to be rotated.
    :param angles: float array, shape=[N], rotated angle.
    :param axis: rotated axis
    :return: float array: same shape as points
    """
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, rot_sin], [zeros, ones, zeros], [-rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, rot_sin, zeros], [-rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, rot_sin], [zeros, -rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)


def rotation_points_single_angle(points, angle, axis=0):
    """
    rotate points by angle along the axis
    :param points: input points
    :param angle: rotated angle
    :param axis: rotated axis
    :return: rotated points
    """
    # points: [N, 3]
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]], dtype=points.dtype)
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array([[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]], dtype=points.dtype)
    elif axis == 0:
        rot_mat_T = np.array([[1, 0, 0], [0, rot_cos, rot_sin], [0, -rot_sin, rot_cos]], dtype=points.dtype)
    else:
        raise ValueError("axis should in range")

    return points @ rot_mat_T


def points_count_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    """
    count points in region bounded boxes
    :param points: input points
    :param rbbox: region bounded boxes
    :param z_axis: voxel axis
    :param origin: list or array or float, origin point relate to smallest point. [0.5, 0.5, 0.5] in lidar.
    :return: [num_polygon] count array.
    """
    rbbox_corners = center_to_corner_box3d(rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=z_axis)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    return points_count_convex_polygon_3d_jit(points[:, :3], surfaces)


def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    """
    find point indices in region bounded boxes
    :param points: input points
    :param rbbox: region bounded boxes
    :param z_axis: voxel axis
    :param origin: list or array or float, origin point relate to smallest point. [0.5, 0.5, 0.5] in lidar.
    :return: point indices that indicating whether belong to one box or not
    """
    rbbox_corners = center_to_corner_box3d(rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=z_axis)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def create_anchors_3d_range(feature_size, anchor_range, sizes=[1.6, 3.9, 1.56],
                            rotations=[0, np.pi / 2], dtype=np.float32):
    """
    create anchors in 3d range
    :param feature_size: list [D, H, W](zyx)
    :param anchor_range: point cloud range for generating anchors
    :param sizes: [N, 3] list of list or array, size of anchors, xyz
    :param rotations: list of rotations for data augmentation
    :param dtype: type of data
    :return: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    anchor_range = np.array(anchor_range, dtype)
    z_centers = np.linspace(anchor_range[2], anchor_range[5], feature_size[0], dtype=dtype)
    y_centers = np.linspace(anchor_range[1], anchor_range[4], feature_size[1], dtype=dtype)
    x_centers = np.linspace(anchor_range[0], anchor_range[3], feature_size[2], dtype=dtype)
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    res = np.transpose(ret, [2, 1, 0, 3, 4, 5])
    return res


def limit_period(val, offset=0.5, period=np.pi):
    """
    get the limit period value
    :param val: the input value
    :param offset: the offset of value
    :param period: the limited period
    :return: limited value
    """
    return val - np.floor(val / period + offset) * period


@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=1.0):
    """
    calculate box iou. note that jit version runs 2x faster than cython in my machine
    :param boxes: (N, 4) ndarray of float
    :param query_boxes: (K, 4) ndarray of float
    :param eps: epsilon to the minimal value
    :return: overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = ((boxes[n, 2] - boxes[n, 0] + eps) *
                          (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def voxel_box_encode(boxes, anchors):
    """
    box encode for voxelnet in lidar
    :param boxes: [N, 7 + ?] x, y, z, w, l, h, r, custom values
    :param anchors: [N, 7] Tensor
    :return:
    """
    # need to convert boxes to z-center format
    box_ndim = anchors.shape[-1]
    cas, cgs = [], []
    if box_ndim > 7:
        xa, ya, za, la, wa, ha, ra, *cas = np.split(anchors, box_ndim, axis=1)
        xg, yg, zg, lg, wg, hg, rg, *cgs = np.split(boxes, box_ndim, axis=1)
    else:
        xa, ya, za, la, wa, ha, ra = np.split(anchors, box_ndim, axis=1)
        xg, yg, zg, lg, wg, hg, rg = np.split(boxes, box_ndim, axis=1)

    for i in range(len(la)):
        la[i] = 1e-3 if la[i] < 1e-3 else la[i]
        wa[i] = 1e-3 if wa[i] < 1e-3 else wa[i]
        ha[i] = 1e-3 if ha[i] < 1e-3 else ha[i]

    diagonal = np.sqrt(la**2 + wa**2)  # 4.3
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha  # 1.6
    cts = [g - a for g, a in zip(cgs, cas)]

    for i in range(len(la)):
        lg[i] = 1e-3 if lg[i] < 1e-3 else lg[i]
        wg[i] = 1e-3 if wg[i] < 1e-3 else wg[i]
        hg[i] = 1e-3 if hg[i] < 1e-3 else hg[i]

    lt = np.log(lg / la)
    wt = np.log(wg / wa)
    ht = np.log(hg / ha)

    rt = rg - ra
    return np.concatenate([xt, yt, zt, lt, wt, ht, rt, *cts], axis=1)


def voxel_box_decode(box_encodings, anchors):
    """
    box decode for pillarnet in lidar
    :param box_encodings: [N, 7] array, normal boxes: x, y, z, w, l, h, r
    :param anchors: [N, 7] array, anchors
    :return: decoded boxes
    """
    # need to convert box_encodings to z-bottom format
    box_ndim = anchors.shape[-1]
    cas, cts = [], []
    if box_ndim > 7:
        xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = np.split(box_encodings, box_ndim, axis=-1)
    else:
        xa, ya, za, wa, la, ha, ra = np.split(anchors, box_ndim, axis=-1)
        xt, yt, zt, wt, lt, ht, rt = np.split(box_encodings, box_ndim, axis=-1)

    diagonal = np.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    lg = np.exp(lt) * la
    wg = np.exp(wt) * wa
    hg = np.exp(ht) * ha
    rg = rt + ra
    cgs = [t + a for t, a in zip(cts, cas)]
    return np.concatenate([xg, yg, zg, wg, lg, hg, rg, *cgs], axis=-1)


@numba.jit(nopython=True)
def distance_similarity(points, qpoints, dist_norm, with_rotation=False, rot_alpha=0.5):
    """
    calculate the distance similarity
    :param points: lidar points
    :param qpoints: query points
    :param dist_norm: distance normalization
    :param with_rotation: whether using rotation
    :param rot_alpha: the rotational factor
    :return: the distance similarity
    """
    N = points.shape[0]
    K = qpoints.shape[0]
    dists = np.zeros((N, K), dtype=points.dtype)
    rot_alpha_1 = 1 - rot_alpha
    for k in range(K):
        for n in range(N):
            if np.abs(points[n, 0] - qpoints[k, 0]) <= dist_norm:
                if np.abs(points[n, 1] - qpoints[k, 1]) <= dist_norm:
                    dist = np.sum((points[n, :2] - qpoints[k, :2])**2)
                    dist_normed = min(dist / dist_norm, dist_norm)
                    if with_rotation:
                        dist_rot = np.abs(np.sin(points[n, -1] - qpoints[k, -1]))
                        dists[n, k] = 1 - rot_alpha_1 * dist_normed - rot_alpha * dist_rot
                    else:
                        dists[n, k] = 1 - dist_normed
    return dists


def camera_to_lidar(points, r_rect, velo2cam):
    """Convert points in camera coordinate to lidar coordinate.

    Args:
        points (np.ndarray, shape=[N, 3]): Points in camera coordinate.
        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.

    Returns:
        np.ndarray, shape=[N, 3]: Points in lidar coordinate.
    """
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    """Covert boxes in camera coordinate to lidar coordinate.

    Args:
        data (np.ndarray, shape=[N, 7]): Boxes in camera coordinate.
        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.

    Returns:
        np.ndarray, shape=[N, 3]: Boxes in lidar coordinate.
    """
    xyz = data[:, 0:3]
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def change_box3d_center(box3d, src, dst):
    dst = np.array(dst, dtype=box3d.dtype)
    src = np.array(src, dtype=box3d.dtype)
    box3d[..., :3] += box3d[..., 3:6] * (dst - src)
