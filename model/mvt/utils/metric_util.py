import torch.nn as nn
import torch
import numpy as np

from .bbox_util import bbox_overlaps_np
from .io_util import imread


def calculate_confusion_matrix(pred, target):
    if isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
        pred = torch.from_numpy(pred)
        target = torch.from_numpy(target)
    elif not (isinstance(pred, torch.Tensor)
              and isinstance(target, torch.Tensor)):
        raise TypeError('pred and target should both be'
                        'torch.Tensor or np.ndarray')
    _, pred_label = pred.topk(1, dim=1)
    num_classes = pred.size(1)
    pred_label = pred_label.view(-1)
    target_label = target.view(-1)
    assert len(pred_label) == len(target_label)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        confusion_matrix[target_label.long(), pred_label.long()] += 1
    return confusion_matrix


def precision(pred, target):
    """Calculate macro-averaged precision according to the prediction and target

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.

    Returns:
        float: The function will return a single float as precision.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(0), min=1)
        res = res.mean().item() * 100
    return res


def recall(pred, target):
    """Calculate macro-averaged recall according to the prediction and target

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.

    Returns:
        float: The function will return a single float as recall.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(1), min=1)
        res = res.mean().item() * 100
    return res


def f1_score(pred, target):
    """Calculate macro-averaged F1 score according to the prediction and target

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.

    Returns:
        float: The function will return a single float as F1 score.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        precision = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(1), min=1)
        recall = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(0), min=1)
        res = 2 * precision * recall / torch.clamp(
            precision + recall, min=1e-20)
        res = torch.where(torch.isnan(res), torch.full_like(res, 0), res)
        res = res.mean().item() * 100
    return res


def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor or np.array): The model prediction, 
            shape (N, num_class)
        target (torch.Tensor or np.array): The target of each 
            prediction, shape (N, )
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """

    if isinstance(topk, list):
        topk = tuple(topk)
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    if type(pred) is np.ndarray:
        pred = torch.from_numpy(pred)
    if type(target) is np.ndarray:
        target = torch.from_numpy(target)

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for _ in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.transpose(0, 1) #.t()  # transpose to shape (maxk, N)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    # correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []

    all_size = pred.size(0)
    for i in range(2, len(pred.shape)):
        all_size = all_size * pred.shape[i]
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / all_size))
    return res[0] if return_single else res


def cal_recalls(all_ious, proposal_nums, thrs):

    img_num = all_ious.shape[0]
    total_gt_num = sum([ious.shape[0] for ious in all_ious])

    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros((ious.shape[0]))
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious

    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)

    return recalls


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.
    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.
    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (
                det_bboxes[:, 3] - det_bboxes[:, 1])
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp

    ious = bbox_overlaps_np(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def tpfp_imagenet(det_bboxes,
                  gt_bboxes,
                  gt_bboxes_ignore=None,
                  default_iou_thr=0.5,
                  area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        default_iou_thr (float): IoU threshold to be considered as matched for
            medium and large bboxes (small ones have special rules).
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp
    # of a certain scale.
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (
                det_bboxes[:, 3] - det_bboxes[:, 1])
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp
    ious = bbox_overlaps_np(det_bboxes, gt_bboxes - 1)
    gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
    iou_thrs = np.minimum((gt_w * gt_h) / ((gt_w + 10.0) * (gt_h + 10.0)),
                          default_iou_thr)
    # sort all detections by scores in descending order
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = gt_w * gt_h
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            max_iou = -1
            matched_gt = -1
            # find best overlapped available gt
            for j in range(num_gts):
                # different from PASCAL VOC: allow finding other gts if the
                # best overlaped ones are already matched by other det bboxes
                if gt_covered[j]:
                    continue
                elif ious[i, j] >= iou_thrs[j] and ious[i, j] > max_iou:
                    max_iou = ious[i, j]
                    matched_gt = j
            # there are 4 cases for a det bbox:
            # 1. it matches a gt, tp = 1, fp = 0
            # 2. it matches an ignored gt, tp = 0, fp = 0
            # 3. it matches no gt and within area range, tp = 0, fp = 1
            # 4. it matches no gt but is beyond area range, tp = 0, fp = 0
            if matched_gt >= 0:
                gt_covered[matched_gt] = 1
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    tp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


class Accuracy(nn.Module):
    """Module to calculate the accuracy."""

    def __init__(self, topk=(1, ), thresh=None):
        """Initialization for computing accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh)


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map.
        label (ndarray): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = imread(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label
