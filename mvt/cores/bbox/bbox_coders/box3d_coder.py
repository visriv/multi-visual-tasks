from abc import ABCMeta
from abc import abstractmethod
import torch
import numpy as np
from torch import stack as tstack

from mvt.utils import bbox3d_util

# general
def torch_to_np_dtype(ttype):
    """convert torch to numpy"""
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float64: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]


# Box
def voxel_box_encode(boxes, anchors):
    """
    box encode for voxel-net
    :param boxes: [N, 7] Tensor, normal boxes: x, y, z, l, w, h, r
    :param anchors: [N, 7] Tensor, anchors
    :param encode_angle_to_vector: whether encoding angle to vector
    :param smooth_dim: whether using smooth dim
    :return: encoded boxes
    """
    box_ndim = anchors.shape[-1]
    cas, cgs = [], []
    if box_ndim > 7:
        xa, ya, za, la, wa, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, lg, wg, hg, rg, *cgs = torch.split(boxes, 1, dim=-1)
    else:
        xa, ya, za, la, wa, ha, ra = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, lg, wg, hg, rg = torch.split(boxes, 1, dim=-1)

    la = torch.clamp(la, 1e-3, 1e3)
    wa = torch.clamp(wa, 1e-3, 1e3)
    ha = torch.clamp(ha, 1e-3, 1e3)
    lg = torch.clamp(la, 1e-3, 1e3)
    wg = torch.clamp(wa, 1e-3, 1e3)
    hg = torch.clamp(ha, 1e-3, 1e3)

    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha
    cts = [g - a for g, a in zip(cgs, cas)]

    lt = torch.log(lg / la)
    wt = torch.log(wg / wa)
    ht = torch.log(hg / ha)

    rt = rg - ra
    return torch.cat([xt, yt, zt, lt, wt, ht, rt, *cts], dim=-1)


def voxel_box_decode(box_encodings, anchors):
    """
    box decode for pillar-net in lidar
    :param box_encodings: [N, 7] Tensor, normal boxes: x, y, z, w, l, h, r
    :param anchors: [N, 7] Tensor, anchors
    :param encode_angle_to_vector: whether encoding angle to vector
    :param smooth_dim: whether using smooth dim
    :return: decoded boxes
    """
    box_ndim = anchors.shape[-1]
    cas, cts = [], []
    if box_ndim > 7:
        xa, ya, za, la, wa, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, lt, wt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)
    else:
        xa, ya, za, la, wa, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, lt, wt, ht, rt = torch.split(box_encodings, 1, dim=-1)

    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za
    lg = torch.exp(lt) * la
    wg = torch.exp(wt) * wa
    hg = torch.exp(ht) * ha
    rg = rt + ra
    cgs = [t + a for t, a in zip(cts, cas)]
    return torch.cat([xg, yg, zg, lg, wg, hg, rg, *cgs], dim=-1)


def corners_nd(dims, origin=0.5):
    """
    generate relative box corners based on length per dim and origin point.
    :param dims: float array, shape=[N, ndim], array of length per dim
    :param origin: list or array or float, origin point relate to smallest point.
    :return: float array, shape=[N, 2 ** ndim, ndim], returned corners.
             point layout example, (2d) x0y0, x0y1, x1y0, x1y1;
             (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
             where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)
    return corners


def corner_to_standup_nd(boxes_corner):
    """
    convert box corners to stand-up box corners
    :param boxes_corner: the given box corners
    :return: stand-up box corners that are along axises
    """
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(torch.min(boxes_corner[:, :, i], dim=1)[0])
    for i in range(ndim):
        standup_boxes.append(torch.max(boxes_corner[:, :, i], dim=1)[0])
    return torch.stack(standup_boxes, dim=1)


def rotation_2d(points, angles):
    """
    rotation 2d points based on origin point clockwise when angle positive.
    :param points: float array, shape=[N, point_size, 2], points to be rotated.
    :param angles: float array, shape=[N], rotation angle.
    :return: float array: same shape as points
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack([tstack([rot_cos, rot_sin]), tstack([-rot_sin, rot_cos])])
    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """
    convert kitti locations, dimensions and angles to corners
    :param centers: float array, shape=[N, 2], locations in kitti label file.
    :param dims: float array, shape=[N, 2], dimensions in kitti label file.
    :param angles: float array, shape=[N], rotation_y in kitti label file.
    :param origin: list or array or float, origin point relate to smallest point.
    :return: float array, shape=[N, 2 ** ndim, ndim], returned corners.
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners


# Box Class
class BoxCoder(object):
    """Abstract base class for box coder."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def code_size(self):
        pass

    def encode(self, boxes, anchors):
        return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        pass

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        pass


class GroundBox3dCoder(BoxCoder):
    """3d box coder for encoding and decoding 3d box information"""
    def __init__(self):
        super().__init__()

    @property
    def code_size(self):
        return 7

    def _encode(self, boxes, anchors):
        return bbox3d_util.voxel_box_encode(boxes, anchors)

    def _decode(self, encodings, anchors):
        return bbox3d_util.voxel_box_decode(encodings, anchors)


class GroundBox3dCoderTorch(GroundBox3dCoder):
    """3d box coder for encoding and decoding 3d box information, defined by pytorch"""
    def encode_torch(self, boxes, anchors):
        return voxel_box_encode(boxes, anchors)

    def decode_torch(self, boxes, anchors):
        return voxel_box_decode(boxes, anchors)
