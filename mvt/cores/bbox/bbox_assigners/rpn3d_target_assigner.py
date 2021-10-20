import numpy as np
import logging
from collections import OrderedDict

from mvt.cores.bbox.bbox_coders.box3d_coder import GroundBox3dCoderTorch
from mvt.cores.anchor.anchor_range3d_generator import AnchorGeneratorRange
from mvt.utils.similarity3d_util import NearestIouSimilarity


def unmap(data, count, inds, fill=0):
    """
    Unmap a subset of item (data) back to the original set of items (of size count)
    :param data: input data
    :param count: the total count of data
    :param inds: the selected indices of input data
    :param fill: filled value
    :return: unmaped data
    """
    if count == len(inds):
        return data
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


# Target Process
def create_target_np(
    all_anchors, 
    gt_boxes, 
    similarity_fn, 
    box_encoding_fn, 
    gt_classes=None,
    matched_threshold=0.6, 
    unmatched_threshold=0.45, 
    box_code_size=7
):
    """
    Modified from FAIR detectron.
    :param all_anchors: [num_of_anchors, box_ndim] float tensor.
    :param gt_boxes: [num_gt_boxes, box_ndim] float tensor.
    :param similarity_fn: a function, accept anchors and gt_boxes, return similarity matrix(such as IoU).
    :param box_encoding_fn: a function, accept gt_boxes and anchors, return box encodings(offsets).
    :param gt_classes: [num_gt_boxes] int tensor. indicate gt classes, must start with 1.
    :param matched_threshold: float, iou greater than matched_threshold will be treated as positives.
    :param unmatched_threshold: float, iou smaller than unmatched_threshold will be treated as negatives.
    :param box_code_size: the size of coded box
    :return: labels, bbox_targets, bboxcreate_target_np_outside_weights
    """
    total_anchors = all_anchors.shape[0]
    anchors = all_anchors
    num_inside = total_anchors

    logger = logging.getLogger(__name__)
    logger.debug('total_anchors: {}'.format(total_anchors))
    logger.debug('inds_inside: {}'.format(num_inside))
    logger.debug('anchors.shape: {}'.format(anchors.shape))

    if gt_classes is None:
        gt_classes = np.ones([gt_boxes.shape[0]], dtype=np.int32)

    labels = np.empty((num_inside, ), dtype=np.int32)
    gt_ids = np.empty((num_inside, ), dtype=np.int32)
    labels.fill(-1)
    gt_ids.fill(-1)
    importance = np.empty((num_inside, ), dtype=np.float32)
    importance.fill(1)
    if len(gt_boxes) > 0:
        # Compute overlaps between the anchors and the gt boxes overlaps
        anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside), anchor_to_gt_argmax]  #
        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, np.arange(anchor_by_gt_overlap.shape[1])]
        # must remove gt which doesn't match any anchor.
        empty_gt_mask = gt_to_anchor_max == 0
        gt_to_anchor_max[empty_gt_mask] = -1
        # Find all anchors that share the max overlap amount (this includes many ties)
        anchors_with_max_overlap = np.where(anchor_by_gt_overlap == gt_to_anchor_max)[0]
        # Fg label: for each gt use anchors with highest overlap (including ties)
        gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
        labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        gt_ids[anchors_with_max_overlap] = gt_inds_force
        # Fg label: above threshold IOU
        pos_inds = anchor_to_gt_max >= matched_threshold
        gt_inds = anchor_to_gt_argmax[pos_inds]
        labels[pos_inds] = gt_classes[gt_inds]
        gt_ids[pos_inds] = gt_inds
        bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
    else:
        bg_inds = np.arange(num_inside)
    fg_inds = np.where(labels > 0)[0]
    fg_max_overlap = None
    if len(gt_boxes) > 0:
        fg_max_overlap = anchor_to_gt_max[fg_inds]
    gt_pos_ids = gt_ids[fg_inds]

    if len(gt_boxes) == 0:
        labels[:] = 0
    else:
        labels[bg_inds] = 0
        # re-enable anchors_with_max_overlap
        labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

    bbox_targets = np.zeros((num_inside, box_code_size), dtype=all_anchors.dtype)
    if len(gt_boxes) > 0:
        # print(anchors[fg_inds, :].shape, gt_boxes[anchor_to_gt_argmax[fg_inds], :].shape)
        # bbox_targets[fg_inds, :] = box_encoding_fn(
        #     anchors[fg_inds, :], gt_boxes[anchor_to_gt_argmax[fg_inds], :])
        bbox_targets[fg_inds, :] = box_encoding_fn(gt_boxes[anchor_to_gt_argmax[fg_inds], :], anchors[fg_inds, :])

    bbox_outside_weights = np.zeros((num_inside, ), dtype=all_anchors.dtype)

    # uniform weighting of examples (given non-uniform sampling)
    bbox_outside_weights[labels > 0] = 1.0

    ret = {
        "labels": labels,
        "bbox_targets": bbox_targets,
        "bbox_outside_weights": bbox_outside_weights,
        "assigned_anchors_overlap": fg_max_overlap,
        "positive_gt_id": gt_pos_ids
    }

    return ret


class RPNTargetAssigner():
    """assign target to the corresponding anchors"""
    def __init__(self, class_settings, anchor_settings):
        """
        initialization
        :param region_similarity_calculators: the calculator for measuring similarity
        """
        self._box_coder = GroundBox3dCoderTorch()
        anchor_generators = []
        for single_anchor in anchor_settings:
            anchor_generator = AnchorGeneratorRange(
                sizes=anchor_settings[single_anchor]["anchor_sizes"],
                anchor_ranges=anchor_settings[single_anchor]["anchor_ranges"],
                rotations=anchor_settings[single_anchor]["anchor_rotations"],
                match_threshold=anchor_settings[single_anchor]["matched_threshold"],
                unmatch_threshold=anchor_settings[single_anchor]["unmatched_threshold"],
                size_name=single_anchor,
                custom_values=[])
            anchor_generators.append(anchor_generator)
        self._anchor_generators = anchor_generators

        classes = []
        for single_class in class_settings:
            classes.append(class_settings[single_class]["class_name"])
        self._classes = classes

        self._sim_calcs = NearestIouSimilarity()
        box_ndims = [a.ndim for a in anchor_generators]
        assert all([e == box_ndims[0] for e in box_ndims])
        self._box_coder.custom_ndim = self.get_custom_ndim()

    @property
    def box_coder(self):
        return self._box_coder

    @property
    def classes(self):
        return self._classes

    @property
    def box_ndim(self):
        return self._anchor_generators[0].ndim

    @property
    def num_anchors_per_location(self):
        num = 0
        for a_generator in self._anchor_generators:
            num += a_generator.num_anchors_per_localization
        return num

    def get_custom_ndim(self):
        return self._anchor_generators[0].custom_ndim

    def assign_all(self, anchors, gt_boxes, gt_classes=None, matched_thresholds=None, unmatched_thresholds=None):
        """
        assign all gt_boxes to the corresponding anchors
        :param anchors: fixed anchors
        :param gt_boxes: ground truth boxes
        :param gt_classes: the ground truth classes
        :param matched_thresholds: list of matched thresholds for classes
        :param unmatched_thresholds: list of unmatched thresholds for classes
        :return:
        """
        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
            return self._sim_calcs.compare(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(boxes, anchors)

        return create_target_np(
            anchors, gt_boxes, similarity_fn, box_encoding_fn,
            gt_classes=gt_classes,
            matched_threshold=matched_thresholds,
            unmatched_threshold=unmatched_thresholds,
            box_code_size=self.box_coder.code_size)

    def generate_anchors(self, feature_map_size):
        """
        generate anchors for region proposal
        :param feature_map_size: the size of feature map for proposing object regions
        :return: generated anchors
        """
        anchors_list = []
        ndim = len(feature_map_size)
        matched_thresholds = [a.match_threshold for a in self._anchor_generators]
        unmatched_thresholds = [a.unmatch_threshold for a in self._anchor_generators]
        match_list, unmatch_list = [], []

        feature_map_sizes = [feature_map_size] * len(self._anchor_generators)

        for anchor_generator, match_thresh, unmatch_thresh, fsize in \
                zip(self._anchor_generators, matched_thresholds, unmatched_thresholds, feature_map_sizes):
            anchors = anchor_generator.generate(fsize)
            anchors = anchors.reshape([*fsize, -1, self.box_ndim])
            anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)
            anchors_list.append(anchors.reshape(-1, self.box_ndim))
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(np.full([num_anchors], unmatch_thresh, anchors.dtype))

        anchors = np.concatenate(anchors_list, axis=0)
        matched_thresholds = np.concatenate(match_list, axis=0)
        unmatched_thresholds = np.concatenate(unmatch_list, axis=0)
        return {
            "anchors": anchors,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds
        }

    def generate_anchors_dict(self, feature_map_size):
        """
        generate the dict of anchors
        :param feature_map_size: the size of feature map for proposing object regions
        :return: dict of generated anchors
        """
        ndim = len(feature_map_size)
        matched_thresholds = [a.match_threshold for a in self._anchor_generators]
        unmatched_thresholds = [a.unmatch_threshold for a in self._anchor_generators]
        match_list, unmatch_list = [], []
        anchors_dict = OrderedDict()
        for a in self._anchor_generators:
            anchors_dict[a.size_name] = {}

        feature_map_sizes = [feature_map_size] * len(self._anchor_generators)
        for anchor_generator, match_thresh, unmatch_thresh, fsize in \
                zip(self._anchor_generators, matched_thresholds, unmatched_thresholds, feature_map_sizes):
            anchors = anchor_generator.generate(fsize)
            anchors = anchors.reshape([*fsize, -1, self.box_ndim])
            anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(np.full([num_anchors], unmatch_thresh, anchors.dtype))
            size_name = anchor_generator.size_name
            anchors_dict[size_name]["anchors"] = anchors.reshape(-1, self.box_ndim)
            anchors_dict[size_name]["matched_thresholds"] = match_list[-1]
            anchors_dict[size_name]["unmatched_thresholds"] = unmatch_list[-1]
        return anchors_dict
