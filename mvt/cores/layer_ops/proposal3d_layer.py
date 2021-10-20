import torch
from torch import nn
import numpy as np
from torchvision.ops import nms
from mvt.cores.bbox.bbox_coders.box3d_coder import (
    center_to_corner_box2d,
    corner_to_standup_nd
)
from mvt.cores.bbox.bbox_transforms import limit_period_torch


def refine_detections(slice_box_preds, slice_cls_scores, slice_dir_labels, slice_gt_boxes=None, slice_gt_labels=None,
                      num_directions=2, iou_threshold=0.1, topk_count=1000, max_count=200):
    """
    Refine classified proposals and filter overlaps and return final detections.
    :param slice_box_preds: [N, 7] in normalized coordinates
    :param slice_cls_scores: [N, num_classes]. Class probabilities
    :param slice_dir_labels: [N,]
    :param score_threshold:
    :param iou_threshold:
    :param max_instances:
    :return: [max_instances, 7]
             [max_instances, ] label_obj
             [max_instances, ] score_obj
             [max_instances, ] label_dir
    """
    class_scores, class_labels = torch.max(slice_cls_scores, dim=-1)
    top_class_scores, topk_indices = torch.topk(class_scores, topk_count, sorted=True)
    top_box_preds = slice_box_preds[topk_indices]
    top_class_labels = class_labels[topk_indices]
    top_dir_labels = slice_dir_labels[topk_indices]

    if top_class_scores.shape[0] != 0:
        boxes_for_nms = top_box_preds[:, [0, 1, 3, 4, 6]]
        box_preds_corners = center_to_corner_box2d(
            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
        boxes_for_nms = corner_to_standup_nd(box_preds_corners)
        selected = nms(boxes_for_nms, top_class_scores, iou_threshold=iou_threshold)
    else:
        selected = []

    pad_gap = max_count - len(selected)
    device = slice_box_preds.device
    if len(selected) > 0:
        selected_boxes = top_box_preds[selected]
        selected_labels = top_class_labels[selected]
        selected_scores = top_class_scores[selected]
        selected_dirlabels = top_dir_labels[selected]

        sel_period = 2 * np.pi / num_directions
        sel_dir_rot = limit_period_torch(selected_boxes[..., 6], 0, sel_period)
        selected_boxes[..., 6] = sel_dir_rot + sel_period * selected_dirlabels.to(selected_boxes.dtype)

        # Pad with zeros if detections < DETECTION_MAX_INSTANCES
        if pad_gap > 0:
            det_boxes = torch.cat(
                [selected_boxes, torch.zeros([pad_gap, 7], dtype=selected_boxes.dtype).to(device)], dim=0)
            det_classids = torch.cat(
                [selected_labels, torch.zeros([pad_gap], dtype=selected_labels.dtype).to(device)], dim=0)
            det_classscores = torch.cat(
                [selected_scores, torch.zeros([pad_gap], dtype=selected_scores.dtype).to(device)], dim=0)
        else:
            det_boxes = selected_boxes[:max_count, :]
            det_classids = selected_labels[:max_count]
            det_classscores = selected_scores[:max_count]
    else:
        det_boxes = torch.zeros([max_count, 7], dtype=top_box_preds.dtype).to(device)
        det_classids = torch.zeros([max_count], dtype=top_class_labels.dtype).to(device)
        det_classscores = torch.zeros([max_count], dtype=top_class_scores.dtype).to(device)

    if slice_gt_boxes is not None:
        top_gt_boxes = slice_gt_boxes[topk_indices]
        top_gt_labels = slice_gt_labels[topk_indices]
        if len(selected) > 0:
            selected_gt_boxes = top_gt_boxes[selected]
            selected_gt_labels = top_gt_labels[selected]
            if pad_gap > 0:
                det_gt_boxes = torch.cat(
                    [selected_gt_boxes, torch.zeros([pad_gap, 7], dtype=selected_gt_boxes.dtype).to(device)], dim=0)
                det_gt_labels = torch.cat(
                    [selected_gt_labels, torch.zeros([pad_gap, 1], dtype=selected_gt_labels.dtype).to(device)], dim=0)
            else:
                det_gt_boxes = selected_gt_boxes[:max_count]
                det_gt_labels = selected_gt_labels[:max_count]
        else:
            det_gt_boxes = torch.zeros([max_count, 7], dtype=slice_gt_boxes.dtype).to(device)
            det_gt_labels = torch.zeros([max_count, 1], dtype=slice_gt_labels.dtype).to(device)
        return det_boxes, det_classids, det_classscores, det_gt_boxes, det_gt_labels
    else:
        return det_boxes, det_classids, det_classscores


class ProposalLayer(nn.Module):
    """
    Receives anchor scores and selects a subset to pass as proposals to the second stage. Filtering is done based on
    anchor scores and non-max suppression to remove overlaps. It also applies bounding box refinement deltas to anchors.
    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates
    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, iou_threshold, topk_count, proposal_count, num_directions):
        super(ProposalLayer, self).__init__()
        self.iou_threshold = iou_threshold
        self.topk_count = topk_count
        self.proposal_count = proposal_count
        self._num_directions = num_directions

    def forward(self, num_class, box_pred, cls_pred, dir_pred, box_targets=None, cls_targets=None):
        """
        Receives anchor scores and selects a subset to pass as proposals tflooro the second stage.
        :param anchors:
        :param cls_pred:
        :param dir_pred:
        :return:
        """
        batch_size = box_pred.shape[0]
        num_class_with_bg = num_class
        
        batch_box_preds = box_pred
        batch_cls_preds = cls_pred.view(batch_size, -1, num_class_with_bg)
        batch_cls_preds = torch.sigmoid(batch_cls_preds)
        batch_dir_preds = torch.max(dir_pred, dim=-1)[1]

        batch_det_boxes = []
        batch_det_classids = []
        batch_det_classscores = []
        batch_gt_boxes = []
        batch_gt_labels = []

        if box_targets is None:
            for box_preds, cls_preds, dir_preds in zip(batch_box_preds, batch_cls_preds, batch_dir_preds):
                det_boxes, det_classids, det_classscores = \
                    refine_detections(box_preds, cls_preds, dir_preds,
                                      num_directions=self._num_directions, iou_threshold=self.iou_threshold,
                                      topk_count=self.topk_count, max_count=self.proposal_count)
                batch_det_boxes.append(det_boxes)
                batch_det_classids.append(det_classids)
                batch_det_classscores.append(det_classscores)

            batch_det_boxes = torch.stack(batch_det_boxes, dim=0)
            batch_det_classids = torch.stack(batch_det_classids, dim=0)
            batch_det_classscores = torch.stack(batch_det_classscores, dim=0)
            return batch_det_boxes, batch_det_classids, batch_det_classscores
        else:
            for box_preds, cls_preds, dir_preds, box_gts, cls_gts in zip(
                    batch_box_preds, batch_cls_preds, batch_dir_preds, box_targets, cls_targets):
                det_boxes, det_classids, det_classscores, gt_boxes, gt_labels = \
                    refine_detections(box_preds, cls_preds, dir_preds, box_gts, cls_gts,
                                      num_directions=self._num_directions, iou_threshold=self.iou_threshold,
                                      topk_count=self.topk_count, max_count=self.proposal_count)
                batch_det_boxes.append(det_boxes)
                batch_det_classids.append(det_classids)
                batch_det_classscores.append(det_classscores)
                batch_gt_boxes.append(gt_boxes)
                batch_gt_labels.append(gt_labels)

            batch_det_boxes = torch.stack(batch_det_boxes, dim=0)
            batch_det_classids = torch.stack(batch_det_classids, dim=0)
            batch_det_classscores = torch.stack(batch_det_classscores, dim=0)
            batch_gt_boxes = torch.stack(batch_gt_boxes, dim=0)
            batch_gt_labels = torch.stack(batch_gt_labels, dim=0)
            return batch_det_boxes, batch_det_classids, batch_det_classscores, batch_gt_boxes, batch_gt_labels
