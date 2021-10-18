import numba
import numpy as np
import random


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points, voxel_size, coors_range, coor_to_voxelidx, max_points=20):
    """
    get the voxel information with reverse kernel by giving points
    :param points: given points
    :param voxel_size: the size of voxel
    :param coors_range: range of coordinates
    :param coor_to_voxelidx: the index for coordinates to voxels
    :param max_points: max number of points per voxel
    :return: voxels[voxel_size,max_points,4], coordinates[voxel_size,3], num_points_per_voxel[voxel_size]
    """
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise reduce performance
    N = points.shape[0]
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    voxel_num = 0

    voxels = []
    coors = []
    num_points_per_voxel = []

    for i in range(N):
        voxel = np.zeros(shape=(max_points, points.shape[-1]), dtype=points.dtype)
        coor = np.zeros(shape=(3,), dtype=np.int32)
        num_points = 0
        failed = False
        for j in range(3):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:  # if c < 0 or c >= grid_size[j]/2:  #
                failed = True
                break

            coor[2 - j] = c

        if failed:
            continue

        # do not limit z axis when achieving features
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            voxels.append(voxel)
            coors.append(coor)
            num_points_per_voxel.append(num_points)

        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx][num] = points[i]
            num_points_per_voxel[voxelidx] += 1
        else:
            rand_num = random.randint(0, num)
            num_points_per_voxel[voxelidx] += 1
            if rand_num < max_points:
                voxels[voxelidx][rand_num] = points[i]

    for i in range(len(num_points_per_voxel)):
        if num_points_per_voxel[i] > max_points:
            num_points_per_voxel[i] = max_points

    return voxels, coors, num_points_per_voxel


def voxel_fine_selection(voxels, coors, num_points_per_voxel, max_points=100, points_dim=4, max_voxels=50000):
    """
    select max number of voxels by order
    :param voxels: voxels
    :param coors: coordinates of voxels
    :param num_points_per_voxel: number of points in each voxel
    :param max_points: max number of points in each voxel
    :param points_dim: the feature dimension for describing points
    :param max_voxels: max number of voxels
    :return: selected voxels with their features, coordinates and counts
    """
    voxel_num = len(voxels)
    voxels_np = np.stack(voxels, axis=0)
    coors_np = np.stack(coors, axis=0)
    num_points_per_voxel = np.array(num_points_per_voxel, dtype=np.int32)

    if voxel_num > max_voxels:
        indices_num = np.argsort(num_points_per_voxel)[voxel_num-max_voxels:]
        voxels_sel = voxels_np[indices_num, :, :]
        coors_sel = coors_np[indices_num, :]
        num_points_sel = num_points_per_voxel[indices_num]
        # voxels_sel = voxels_np[:max_voxels, :, :]
        # coors_sel = coors_np[:max_voxels, :]
        # num_points_sel = num_points_per_voxel[:max_voxels]
    elif voxel_num == max_voxels:
        voxels_sel = voxels_np
        coors_sel = coors_np
        num_points_sel = num_points_per_voxel
    else:
        voxels_sel = np.concatenate(
            [voxels_np, np.zeros([max_voxels - voxel_num, max_points, points_dim], dtype=np.float32)], axis=0)
        coor_new = np.zeros([max_voxels - voxel_num, 3], dtype=np.int32)
        coors_sel = np.concatenate([coors_np, coor_new], axis=0)
        num_points_sel = np.concatenate(
            [num_points_per_voxel, np.zeros([max_voxels - voxel_num], dtype=np.int32)], axis=0)

    return voxels_sel, coors_sel, num_points_sel


def points_to_voxel(points, voxel_size, coors_range, max_points=30, max_voxels=20000):
    """
    convert points(N, >=3) to voxels. This version calculate everything in one loop.
    :param points: [N, ndim] float tensor. points[:, :3] contain xyz points and
           points[:, 3:] contain other information such as reflectivity.
    :param voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
    :param coors_range: [6] list/tuple or array, float. indicate voxel range. format: xyzxyz, minmax
    :param max_points: int. indicate maximum points contained in a voxel.
    :param max_voxels: int. indicate maximum voxels this function create.
    :return: voxels[voxel_size,max_points,4], coordinates[voxel_size,3], num_points_per_voxel[voxel_size]
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)

    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)

    voxels, coors, num_points = _points_to_voxel_reverse_kernel(
        points, voxel_size, coors_range, coor_to_voxelidx, max_points)

    voxels_randsel, coors_randsel, num_points_randsel = voxel_fine_selection(
        voxels, coors, num_points, max_points, points.shape[-1], max_voxels)

    return voxels_randsel, coors_randsel, num_points_randsel


def square_dis(x, y):
    return (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2])


def fps_sampling(voxels, num_points_randsel, fps_num_list):
    num_voxels = len(voxels)
    point_dim = voxels.shape[-1]
    voxels_1 = np.zeros(shape=(num_voxels, fps_num_list[0], point_dim), dtype=voxels.dtype)
    voxels_2 = np.zeros(shape=(num_voxels, fps_num_list[1], point_dim), dtype=voxels.dtype)

    for i in range(num_voxels):
        voxels_2[i, 0] = voxels[i, 0]  # selected point
        voxels_1[i, 0] = voxels[i, 0]  # selected point
        num_1 = fps_num_list[0] if num_points_randsel[i] > fps_num_list[0] else num_points_randsel[i]
        num_2 = (num_1 + 1)//2
        remain_points = []
        for j in range(num_points_randsel[i]-1):
            remain_points.append(voxels[i, j+1])

        for j in range(num_1-1):
            # Calculate the min distance for each remain points
            remain_dis = []
            for k in range(len(remain_points)):
                min_dis = 1e9
                for n in range(j+1):
                    p_dis = (voxels_1[i, n, 0]-remain_points[k][0])*(voxels_1[i, n, 0]-remain_points[k][0]) \
                            + (voxels_1[i, n, 1]-remain_points[k][1])*(voxels_1[i, n, 1]-remain_points[k][1]) \
                            + (voxels_1[i, n, 2]-remain_points[k][2])*(voxels_1[i, n, 2]-remain_points[k][2])
                    if p_dis < min_dis:
                        min_dis = p_dis
                remain_dis.append(min_dis)
            max_ind = np.argmax(np.array(remain_dis))
            voxels_1[i, j] = remain_points[max_ind]
            if j < num_2:
                voxels_2[i, j] = remain_points[max_ind]

            remain_points.pop(max_ind)

    return voxels_1, voxels_2


def sampling_points_to_voxel(points, voxel_size, coors_range, max_points=30, max_voxels=20000, sample_num_list=[10, 5]):
    """
    convert points(N, >=3) to voxels. This version calculate everything in one loop.
    :param points: [N, ndim] float tensor. points[:, :3] contain xyz points and
           points[:, 3:] contain other information such as reflectivity.
    :param voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
    :param coors_range: [6] list/tuple or array, float. indicate voxel range. format: xyzxyz, minmax
    :param max_points: int. indicate maximum points contained in a voxel.
    :param max_voxels: int. indicate maximum voxels this function create.
    :return: voxels[voxel_size,max_points,4], coordinates[voxel_size,3], num_points_per_voxel[voxel_size]
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)

    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)

    voxels, coors, num_points = _points_to_voxel_reverse_kernel(
        points, voxel_size, coors_range, coor_to_voxelidx, max_points)

    voxels_randsel, coors_randsel, num_points_randsel = voxel_fine_selection(
        voxels, coors, num_points, max_points, points.shape[-1], max_voxels)

    voxels_1, voxels_2 = fps_sampling(voxels_randsel, num_points_randsel, sample_num_list)

    return voxels_1, voxels_2, coors_randsel, num_points_randsel


def rand_sampling(voxels, num_points, sample_num):
    num_voxels = len(voxels)
    max_num_pt = voxels.shape[-2]
    point_dim = voxels.shape[-1]
    sample_ratio = 1.0 * sample_num/max_num_pt
    voxels_2 = np.zeros(shape=(num_voxels, sample_num, point_dim), dtype=voxels.dtype)

    for i in range(num_voxels):
        num_2 = np.ceil(num_points[i] * sample_ratio).astype(np.int64)
        voxels_2[i, :num_2, :] = voxels[i, :num_2, :]

    return voxels_2


def rand_sampling_points_to_voxel(points, voxel_size, coors_range, max_voxels=20000, sample_num_list=[10, 5]):
    """
    convert points(N, >=3) to voxels. This version calculate everything in one loop.
    :param points: [N, ndim] float tensor. points[:, :3] contain xyz points and
           points[:, 3:] contain other information such as reflectivity.
    :param voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
    :param coors_range: [6] list/tuple or array, float. indicate voxel range. format: xyzxyz, minmax
    :param max_points: int. indicate maximum points contained in a voxel.
    :param max_voxels: int. indicate maximum voxels this function create.
    :return: voxels[voxel_size,max_points,4], coordinates[voxel_size,3], num_points_per_voxel[voxel_size]
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)

    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)

    voxels, coors, num_points = _points_to_voxel_reverse_kernel(
        points, voxel_size, coors_range, coor_to_voxelidx, sample_num_list[0])

    voxels_1, coors_randsel, num_points_randsel = voxel_fine_selection(
        voxels, coors, num_points, sample_num_list[0], points.shape[-1], max_voxels)

    voxels_2 = rand_sampling(voxels_1, num_points_randsel, sample_num_list[1])

    return voxels_1, voxels_2, coors_randsel, num_points_randsel


@numba.jit(nopython=True)
def _dyn_points_to_voxel_reverse_kernel(points, voxel_size, coors_range):
    """
    get the voxel information with reverse kernel by giving points
    :param points: given points
    :param voxel_size: the size of voxel
    :param coors_range: range of coordinates
    :return: point_to_voxelidx[N, 3]
    """
    # put all computations to one loop. we shouldn't create large array in main jit code, otherwise reduce performance
    N = points.shape[0]
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    valid_points = []
    point_to_voxelidx = []

    for i in range(N):
        coor = np.zeros(shape=(3,), dtype=np.int32)
        failed = False
        for j in range(3):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:  # if c < 0 or c >= grid_size[j]/2:  #
                failed = True
                break
            coor[2 - j] = c
        if failed:
            continue

        valid_points.append(points[i])
        point_to_voxelidx.append(coor)

    return valid_points, point_to_voxelidx


def dyn_points_to_voxel(points, voxel_size, coors_range, max_num_points=262144):
    """
    :param points: [N, ndim] float tensor.
    :param voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
    :param coors_range: [6] list/tuple or array, float. indicate voxel range. format: xyzxyz, minmax
    :param max_num_points: int.
    :return: coordinates[N, 3]
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)

    valid_points, point_to_voxelidx = _dyn_points_to_voxel_reverse_kernel(points, voxel_size, coors_range)

    valid_points = np.stack(valid_points, axis=0)
    point_to_voxelidx = np.stack(point_to_voxelidx, axis=0)

    valid_points_num = valid_points.shape[0]

    if valid_points_num >= max_num_points:
        valid_points = valid_points[:max_num_points]
        point_to_voxelidx = point_to_voxelidx[:max_num_points]
    else:
        points_dim = points.shape[1]
        valid_points = np.concatenate([valid_points, np.zeros([max_num_points - valid_points_num, points_dim])], axis=0)
        point_to_voxelidx = np.concatenate([point_to_voxelidx, np.zeros([max_num_points - valid_points_num, 3])], axis=0)

    return valid_points, point_to_voxelidx
