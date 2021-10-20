import torch

from mvt.blocks.block_builder import build_backbone, build_neck, build_head, build_loss
from mvt.cores.bbox.bbox_assigners.rpn3d_target_assigner import RPNTargetAssigner
from mvt.cores.layer_ops.proposal3d_layer import ProposalLayer
from ..model_builder import DETECTORS
from .base_3d_detector import Base3DDetector


def prepare_loss_weights(labels, pos_cls_weight=1.0, neg_cls_weight=1.0, dtype=torch.float32):
    """
    get cls_weights and reg_weights from labels.
    :param labels: ground truth labels
    :param pos_cls_weight: positive classification weight
    :param neg_cls_weight: negative classification weight
    :param dtype: the type of data
    :return: the weights of classification and localization with the selected labels
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)

    pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
    reg_weights /= torch.clamp(pos_normalizer, min=1.0)
    cls_weights /= torch.clamp(pos_normalizer, min=1.0)

    return cls_weights, reg_weights, cared


@DETECTORS.register_module()
class PillarRPN(Base3DDetector):
    def __init__(self, cfg):
        super(Base3DDetector, self).__init__()
        self.type = cfg.TYPE
        self.backbone = build_backbone(cfg.BACKBONE)
        self.neck = build_neck(cfg.NECK)

        self.rpn_head = build_head(cfg.RPN_HEAD)
        self.target_assigner = RPNTargetAssigner(
            cfg.EXTEND.class_settings,
            cfg.EXTEND.anchor_settings
        )
        self._box_coder = self.target_assigner.box_coder
        self.compute_loss = build_loss(cfg.LOSS)

        if "PRETRAINED_MODEL_PATH" in cfg:
            if cfg.PRETRAINED_MODEL_PATH != "":
                self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
            else:
                self.init_weights()
        else:
            self.init_weights()

        self.proposal = ProposalLayer(
            cfg.EXTEND.nms_iou_threshold,
            cfg.EXTEND.nms_pre_max_size,
            cfg.EXTEND.nms_post_max_size,
            cfg.EXTEND.num_direction_bins
        )
        
        self._num_class = cfg.RPN_HEAD.num_classes
        self.num_point_features = cfg.EXTEND.num_point_features
        self.post_center_limit_range = cfg.EXTEND.post_center_limit_range
        self.det_score_threshold = cfg.EXTEND.det_score_threshold

    def init_weights(self, pretrained=None):
        super(Base3DDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.rpn_head.init_weights()

    def forward(self, img, label, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, data_item and data_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, data_item and data_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test.
        """
        if return_loss:
            return self.forward_train(img, label, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

    def forward_train(
        self, 
        voxels, 
        num_points, 
        coordinates, 
        anchors,
        labels,
        reg_targets,
        **kwargs
    ):
        """Forward computation during training."""
        batch_size_dev = anchors.shape[0]
        batch_anchors = anchors.view(batch_size_dev, -1, anchors.shape[-1])

        voxel_features = self.backbone(voxels, num_points, coordinates)
        spatial_features = self.neck(voxel_features, coordinates, batch_size_dev)
        pillar_features = spatial_features.squeeze(-3)

        preds_dict = self.rpn_head(pillar_features)

        box_preds = preds_dict["box_preds"].view(batch_size_dev, -1, self._box_coder.code_size)
        cls_preds = preds_dict["cls_preds"]
        dir_preds = preds_dict["dir_cls_preds"].view(batch_size_dev, -1, 2)

        cls_weights, reg_weights, cared = prepare_loss_weights(labels, dtype=voxels.dtype)
        cls_targets = labels * cared.type_as(labels)
        cls_targets = cls_targets.unsqueeze(-1)
        reg_targets = reg_targets.view(batch_size_dev, -1, self._box_coder.code_size)

        loc_loss, cls_loss, dir_loss = self.compute_loss(
            cls_preds,
            box_preds,
            dir_preds,
            cls_targets,
            reg_targets,
            cls_weights,
            reg_weights,
            labels,
            batch_anchors
        )

        return {
            "cls_loss": cls_loss,
            "loc_loss": loc_loss,
            "dir_loss": dir_loss
        }

    def simple_test(self, voxels, num_points, coordinates, anchors, **kwargs):
        batch_size_dev = anchors.shape[0]
        batch_anchors = anchors.view(batch_size_dev, -1, anchors.shape[-1])

        voxel_features = self.backbone(voxels, num_points, coordinates)
        spatial_features = self.neck(voxel_features, coordinates, batch_size_dev)
        pillar_features = spatial_features.squeeze(-3)

        preds_dict = self.rpn_head(pillar_features)

        box_preds = preds_dict["box_preds"].view(batch_size_dev, -1, self._box_coder.code_size)
        decode_box_preds = self._box_coder.decode_torch(box_preds, batch_anchors)
        cls_preds = preds_dict["cls_preds"]
        dir_preds = preds_dict["dir_cls_preds"].view(batch_size_dev, -1, 2)

        det_boxes, det_labels, det_scores = self.proposal(self._num_class, decode_box_preds, cls_preds, dir_preds)
        return {
            "det_boxes": det_boxes,
            "det_labels": det_labels,
            "det_scores": det_scores
        }
