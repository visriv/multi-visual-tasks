import warnings
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode
from collections import OrderedDict

from .base_det_head import BaseDetHead
from .dense_test_mixins import BBoxTestMixin
from ..block_builder import HEADS, build_loss
from mtl.cores.layer_ops import brick as vn_layer
from mtl.cores.ops import ConvModule, multiclass_nms
from mtl.utils.fp16_util import force_fp32
from mtl.utils.init_util import normal_init
from mtl.cores.anchor import images_to_levels
from mtl.cores.core_anchor import build_anchor_generator
from mtl.cores.core_bbox import build_assigner, build_bbox_coder, build_sampler
from mtl.utils.misc_util import multi_apply


@HEADS.register_module()
class YOLOV3Head(BaseDetHead, BBoxTestMixin):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (dict): Config of classification loss.
        loss_conf (dict): Config of confidence loss.
        loss_xy (dict): Config of xy coordinate loss.
        loss_wh (dict): Config of wh coordinate loss.
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 out_channels=(1024, 512, 256),
                 anchor_generator=dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                     strides=[32, 16, 8]),
                 bbox_coder=dict(type='YOLOBBoxCoder'),
                 featmap_strides=[32, 16, 8],
                 one_hot_smoother=0.,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_conf=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_xy=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_wh=dict(type='MSELoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(YOLOV3Head, self).__init__()

        # Check params
        assert (len(in_channels) == len(out_channels) == len(featmap_strides))

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            if isinstance(self.train_cfg, CfgNode):
                self.assigner = build_assigner(self.train_cfg.assigner)
                if hasattr(self.train_cfg, 'sampler'):
                    sampler_cfg = self.train_cfg.sampler
                else:
                    sampler_cfg = dict(type='PseudoSampler')
            else:
                self.assigner = build_assigner(self.train_cfg['assigner'])
                if 'sample' in self.train_cfg:
                    sampler_cfg = self.train_cfg['sampler']
                else:
                    sampler_cfg = dict(type='PseudoSampler')               
            
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.anchor_generator = build_anchor_generator(anchor_generator)

        self.loss_cls = build_loss(loss_cls)
        self.loss_conf = build_loss(loss_conf)
        self.loss_xy = build_loss(loss_xy)
        self.loss_wh = build_loss(loss_wh)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        assert len(
            self.anchor_generator.num_base_anchors) == len(featmap_strides)
        self._init_layers()

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):

        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_bridge = ConvModule(
                self.in_channels[i],
                self.out_channels[i],
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            conv_pred = nn.Conv2d(self.out_channels[i],
                                  self.num_anchors * self.num_attrib, 1)

            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize weights of the head."""

        for m in self.convs_pred:
            normal_init(m, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)

        return tuple(pred_maps),

    @force_fp32(apply_to=('pred_maps', ))
    def get_bboxes(self,
                   pred_maps,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """

        result_list = []
        num_levels = len(pred_maps)
        for img_id in range(len(img_metas)):
            pred_maps_list = [
                pred_maps[i][img_id].detach() for i in range(num_levels)
            ]
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(pred_maps_list, scale_factor,
                                                cfg, rescale, with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           pred_maps_list,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            pred_maps_list (list[Tensor]): Prediction maps for different scales
                of each single image in the batch.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(pred_maps_list) == self.num_levels
        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        num_levels = len(pred_maps_list)
        featmap_sizes = [
            pred_maps_list[i].shape[-2:] for i in range(num_levels)
        ]
        multi_lvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, pred_maps_list[0][0].device)
        for i in range(self.num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]

            # (h, w, num_anchors*num_attrib) -> (h*w*num_anchors, num_attrib)
            pred_map = pred_map.permute(1, 2, 0).reshape(-1, self.num_attrib)

            pred_map[..., :2] = torch.sigmoid(pred_map[..., :2])
            bbox_pred = self.bbox_coder.decode(multi_lvl_anchors[i],
                                               pred_map[..., :4], stride)
            # conf and cls
            conf_pred = torch.sigmoid(pred_map[..., 4]).view(-1)
            cls_pred = torch.sigmoid(pred_map[..., 5:]).view(
                -1, self.num_classes)  # Cls pred one-hot.
            
            # Filtering out all predictions with conf < conf_thr
            # Get top-k prediction
            if not torch.onnx.is_in_onnx_export():
                conf_thr = cfg.get('conf_thr', -1)
                conf_inds = conf_pred.ge(conf_thr).nonzero().flatten()
                bbox_pred = bbox_pred[conf_inds, :]
                cls_pred = cls_pred[conf_inds, :]
                conf_pred = conf_pred[conf_inds]
                
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < conf_pred.size(0):
                _, topk_inds = torch.topk(conf_pred, nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_pred = cls_pred[topk_inds, :]
                conf_pred = conf_pred[topk_inds]

            # Save the result of current scale
            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)
        
        # Merge the results of different scales together
        multi_lvl_bboxes = torch.cat(multi_lvl_bboxes)
        multi_lvl_cls_scores = torch.cat(multi_lvl_cls_scores)
        multi_lvl_conf_scores = torch.cat(multi_lvl_conf_scores)

        if with_nms and (multi_lvl_conf_scores.size(0) == 0):
            return torch.zeros((0, 5)), torch.zeros((0, ))

        if rescale:
            multi_lvl_bboxes /= multi_lvl_bboxes.new_tensor(scale_factor)

        # the class_id for background is num_classes. i.e., the last column.
        padding = multi_lvl_cls_scores.new_zeros(multi_lvl_cls_scores.shape[0], 1)
        multi_lvl_cls_scores = torch.cat([multi_lvl_cls_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                multi_lvl_bboxes,
                multi_lvl_cls_scores,
                cfg['score_thr'],
                cfg['nms'],
                cfg['max_per_img'],
                score_factors=multi_lvl_conf_scores)
            return det_bboxes, det_labels
        else:
            return (multi_lvl_bboxes, multi_lvl_cls_scores, multi_lvl_conf_scores)

    @force_fp32(apply_to=('pred_maps', ))
    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        num_imgs = len(img_metas)
        device = pred_maps[0][0].device

        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        responsible_flag_list = []
        for img_id in range(len(img_metas)):
            responsible_flag_list.append(
                self.anchor_generator.responsible_flags(
                    featmap_sizes, gt_bboxes[img_id], device))

        target_maps_list, neg_maps_list = self.get_targets(
            anchor_list, responsible_flag_list, gt_bboxes, gt_labels)

        losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(
            self.loss_single, pred_maps, target_maps_list, neg_maps_list)

        return dict(
            loss_cls=losses_cls,
            loss_conf=losses_conf,
            loss_xy=losses_xy,
            loss_wh=losses_wh)

    def loss_single(self, pred_map, target_map, neg_map):
        """Compute loss of a single image from a batch.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            target_map (Tensor): The Ground-Truth target for a single level.
            neg_map (Tensor): The negative masks for a single level.

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(num_imgs, -1, self.num_attrib)
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 4]
        pos_and_neg_mask = neg_mask + pos_mask
        pos_mask = pos_mask.unsqueeze(dim=-1)
        if torch.max(pos_and_neg_mask) > 1.:
            warnings.warn('There is overlap between pos and neg sample.')
            pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

        pred_xy = pred_map[..., :2]
        pred_wh = pred_map[..., 2:4]
        pred_conf = pred_map[..., 4]
        pred_label = pred_map[..., 5:]

        target_xy = target_map[..., :2]
        target_wh = target_map[..., 2:4]
        target_conf = target_map[..., 4]
        target_label = target_map[..., 5:]

        loss_cls = self.loss_cls(pred_label, target_label, weight=pos_mask)
        loss_conf = self.loss_conf(
            pred_conf, target_conf, weight=pos_and_neg_mask)
        loss_xy = self.loss_xy(pred_xy, target_xy, weight=pos_mask)
        loss_wh = self.loss_wh(pred_wh, target_wh, weight=pos_mask)

        return loss_cls, loss_conf, loss_xy, loss_wh

    def get_targets(self, anchor_list, responsible_flag_list, gt_bboxes_list,
                    gt_labels_list):
        """Compute target maps for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi level responsible
                flags of each image. Each element is a tensor of shape
                (num_total_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """

        num_imgs = len(anchor_list)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        results = multi_apply(self._get_targets_single, anchor_list,
                              responsible_flag_list, gt_bboxes_list,
                              gt_labels_list)

        all_target_maps, all_neg_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)

        return target_maps_list, neg_maps_list

    def _get_targets_single(self, anchors, responsible_flags, gt_bboxes,
                            gt_labels):
        """Generate matching bounding box prior and converted GT.
        Args:
            anchors (list[Tensor]): Multi-level anchors of the image.
            responsible_flags (list[Tensor]): Multi-level responsible flags of
                anchors
            gt_bboxes (Tensor): Ground truth bboxes of single image.
            gt_labels (Tensor): Ground truth labels of single image.

        Returns:
            tuple:
                target_map (Tensor): Predication target map of each
                    scale level, shape (num_total_anchors,
                    5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """

        anchor_strides = []
        for i in range(len(anchors)):
            anchor_strides.append(
                torch.tensor(self.featmap_strides[i],
                             device=gt_bboxes.device).repeat(len(anchors[i])))
        concat_anchors = torch.cat(anchors)
        concat_responsible_flags = torch.cat(responsible_flags)

        anchor_strides = torch.cat(anchor_strides)
        assert len(anchor_strides) == len(concat_anchors) == \
               len(concat_responsible_flags)
        assign_result = self.assigner.assign(concat_anchors,
                                             concat_responsible_flags,
                                             gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, concat_anchors,
                                              gt_bboxes)

        target_map = concat_anchors.new_zeros(
            concat_anchors.size(0), self.num_attrib)

        target_map[sampling_result.pos_inds, :4] = self.bbox_coder.encode(
            sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes,
            anchor_strides[sampling_result.pos_inds])

        target_map[sampling_result.pos_inds, 4] = 1

        gt_labels_one_hot = F.one_hot(
            gt_labels, num_classes=self.num_classes).float()
        if self.one_hot_smoother != 0:  # label smooth
            gt_labels_one_hot = gt_labels_one_hot * (
                1 - self.one_hot_smoother
            ) + self.one_hot_smoother / self.num_classes
        target_map[sampling_result.pos_inds, 5:] = gt_labels_one_hot[
            sampling_result.pos_assigned_gt_inds]

        neg_map = concat_anchors.new_zeros(
            concat_anchors.size(0), dtype=torch.uint8)
        neg_map[sampling_result.neg_inds] = 1

        return target_map, neg_map

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """

        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)


@HEADS.register_module()
class TinyYOLOV4Head(YOLOV3Head):

    def _init_layers(self):
        head = [
            OrderedDict([
                ('10_max', nn.MaxPool2d(2, 2)),
                ('11_conv', vn_layer.Conv2dBatchLeaky(self.in_channels[0], self.in_channels[0], 3, 1)),
                ('12_conv', vn_layer.Conv2dBatchLeaky(self.in_channels[0], self.out_channels[0], 1, 1)),
            ]),

            OrderedDict([
                ('13_conv', vn_layer.Conv2dBatchLeaky(self.in_channels[1], self.in_channels[0], 3, 1)),
                ('14_conv', nn.Conv2d(self.in_channels[0], self.num_anchors * self.num_attrib, 1)),
            ]),

            OrderedDict([
                ('15_convbatch', vn_layer.Conv2dBatchLeaky(self.in_channels[1], self.out_channels[1], 1, 1)),
                ('16_upsample', nn.Upsample(scale_factor=2)),
            ]),

            OrderedDict([
                ('17_convbatch', vn_layer.Conv2dBatchLeaky(self.out_channels[0]+self.out_channels[1], 256, 3, 1)),
                ('18_conv', nn.Conv2d(256, self.num_anchors * self.num_attrib, 1)),
            ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in head])
    
    def init_weights(self):
        """Initialize weights of the head."""
        pass

    def forward(self, feats):
        stem, extra_x = feats
        stage0 = self.layers[0](stem)
        head0 = self.layers[1](stage0)
        stage1 = self.layers[2](stage0)
        stage2 = torch.cat((stage1, extra_x), dim=1)
        head1 = self.layers[3](stage2)
        head = [head0, head1]
        return tuple(head),


def _make_divisible(x, divisor, width_multiple):
    return math.ceil(x * width_multiple / divisor) * divisor


def _make_round(x, depth_multiple=1.0):
    return max(round(x * depth_multiple), 1) if x > 1 else x


def make_divisible(divisor, width_multiple=1.0):
    return functools.partial(_make_divisible, divisor=divisor, width_multiple=width_multiple)


def make_round(depth_multiple=1.0):
    return functools.partial(_make_round, depth_multiple=depth_multiple)


@HEADS.register_module()
class YOLOV5Head(BaseDetHead, BBoxTestMixin):
    # dw_ratios = [depth_multiple, width_multiple]

    def __init__(self,
                 num_classes,
                 in_channels,
                 dw_ratios=[1.0, 1.0],
                 anchor_generator=dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[[116, 90], [156, 198], [373, 326]],
                                 [[30, 61], [62, 45], [59, 119]],
                                 [[10, 13], [16, 30], [33, 23]]],
                     strides=[32, 16, 8]),
                 bbox_coder=dict(type='YOLOBBoxCoder'),
                 featmap_strides=[32, 16, 8],
                 one_hot_smoother=0.,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_conf=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_xy=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_wh=dict(type='MSELoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(YOLOV5Head, self).__init__()

        # Check params
        assert (len(in_channels) == len(featmap_strides))

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dw_ratios = dw_ratios
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            if isinstance(self.train_cfg, CfgNode):
                self.assigner = build_assigner(self.train_cfg.assigner)
                if hasattr(self.train_cfg, 'sampler'):
                    sampler_cfg = self.train_cfg.sampler
                else:
                    sampler_cfg = dict(type='PseudoSampler')
            else:
                self.assigner = build_assigner(self.train_cfg['assigner'])
                if 'sample' in self.train_cfg:
                    sampler_cfg = self.train_cfg['sampler']
                else:
                    sampler_cfg = dict(type='PseudoSampler')               
            
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.anchor_generator = build_anchor_generator(anchor_generator)

        self.loss_cls = build_loss(loss_cls)
        self.loss_conf = build_loss(loss_conf)
        self.loss_xy = build_loss(loss_xy)
        self.loss_wh = build_loss(loss_wh)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        assert len(
            self.anchor_generator.num_base_anchors) == len(featmap_strides)
        self._init_layers()

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):
        model = []

        make_div8_fun = make_divisible(8, self.dw_ratios[1])
        make_round_fun = make_round(self.dw_ratios[0])

        conv1 = vn_layer.Conv(make_div8_fun(1024), make_div8_fun(512))
        model.append(conv1)  # 0
        up1 = nn.Upsample(scale_factor=2)
        model.append(up1)  # 1
        cont1 = vn_layer.Concat()
        model.append(cont1)  # 2
        bsp1 = vn_layer.BottleneckCSP(
            make_div8_fun(512) + make_div8_fun(self.in_channels[0]), 
            make_div8_fun(512), make_round_fun(3), shortcut=False)
        model.append(bsp1)  # 3

        conv2 = vn_layer.Conv(make_div8_fun(512), make_div8_fun(256))
        model.append(conv2)  # 4
        up2 = nn.Upsample(scale_factor=2)
        model.append(up2)  # 5
        cont2 = vn_layer.Concat()
        model.append(cont2)  # 6
        bsp2 = vn_layer.BottleneckCSP(
            make_div8_fun(256) + make_div8_fun(self.in_channels[1]), 
            make_div8_fun(256), make_round_fun(3), shortcut=False)
        model.append(bsp2)  # 7

        conv3 = vn_layer.Conv(make_div8_fun(256), make_div8_fun(256), k=3, s=2)
        model.append(conv3)  # 8
        cont3 = vn_layer.Concat()
        model.append(cont3)  # 9
        bsp3 = vn_layer.BottleneckCSP(
            make_div8_fun(256) + make_div8_fun(256), 
            make_div8_fun(512), make_round_fun(3), shortcut=False)
        model.append(bsp3)  # 10

        conv4 = vn_layer.Conv(
            make_div8_fun(512), make_div8_fun(512), k=3, s=2)
        model.append(conv4)  # 11
        cont4 = vn_layer.Concat()
        model.append(cont4)  # 12
        bsp4 = vn_layer.BottleneckCSP(
            make_div8_fun(1024), make_div8_fun(1024), 
            make_round_fun(3), shortcut=False)
        model.append(bsp4)  # 13

        self.det = nn.Sequential(*model)
        self.head = nn.Sequential(
            nn.Conv2d(make_div8_fun(256), self.num_attrib * self.num_levels, 1),
            nn.Conv2d(make_div8_fun(512), self.num_attrib * self.num_levels, 1),
            nn.Conv2d(make_div8_fun(1024), self.num_attrib * self.num_levels, 1),
        )
    
    def init_weights(self):
        """Initialize weights of the head."""
        pass

    def forward(self, feats):
        large_feat, inter_feat, small_feat = feats

        small_feat = self.det[0](small_feat)
        x = self.det[1](small_feat)
        x = self.det[2]([x, inter_feat])
        x = self.det[3](x)
        inter_feat = self.det[4](x)

        x = self.det[5](inter_feat)
        x = self.det[6]([x, large_feat])
        x = self.det[7](x)  # 128
        out0 = self.head[0](x)  # first output

        x = self.det[8](x)
        x = self.det[9]([x, inter_feat])
        x = self.det[10](x)  #
        out1 = self.head[1](x)  # second output

        x = self.det[11](x)
        x = self.det[12]([x, small_feat])
        x = self.det[13](x)  # 256
        out2 = self.head[2](x)  # third output

        return tuple([out2, out1, out0]),  

    @force_fp32(apply_to=('pred_maps', ))
    def get_bboxes(self,
                   pred_maps,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        result_list = []
        num_levels = len(pred_maps)
        for img_id in range(len(img_metas)):
            pred_maps_list = [
                pred_maps[i][img_id].detach() for i in range(num_levels)
            ]
            scale_factor = img_metas[img_id]['scale_factor']
            if 'pad_param' in img_metas[img_id]:
                pad_param = img_metas[img_id]['pad_param']
            else:
                pad_param = None
            proposals = self._get_bboxes_single(pred_maps_list, scale_factor,
                                                cfg, rescale, pad_param, with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           pred_maps_list,
                           scale_factor,
                           cfg,
                           rescale=False,
                           pad_param=None,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            pred_maps_list (list[Tensor]): Prediction maps for different scales
                of each single image in the batch.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """

        cfg = self.test_cfg if cfg is None else cfg
        assert len(pred_maps_list) == self.num_levels
        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        num_levels = len(pred_maps_list)
        featmap_sizes = [
            pred_maps_list[i].shape[-2:] for i in range(num_levels)
        ]
        multi_lvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, pred_maps_list[0][0].device)
        for i in range(self.num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]

            # (h, w, num_anchors*num_attrib) -> (h*w*num_anchors, num_attrib)
            pred_map = pred_map.permute(1, 2, 0).reshape(-1, self.num_attrib)

            pred_map[..., :2] = torch.sigmoid(pred_map[..., :2])
            bbox_pred = self.bbox_coder.decode(multi_lvl_anchors[i],
                                               pred_map[..., :4], stride)
            # conf and cls
            conf_pred = torch.sigmoid(pred_map[..., 4]).view(-1)
            cls_pred = torch.sigmoid(pred_map[..., 5:]).view(
                -1, self.num_classes)  # Cls pred one-hot.
            
            # Filtering out all predictions with conf < conf_thr
            # Get top-k prediction
            if not torch.onnx.is_in_onnx_export():
                conf_thr = cfg.get('conf_thr', -1)
                conf_inds = conf_pred.ge(conf_thr).nonzero().flatten()
                bbox_pred = bbox_pred[conf_inds, :]
                cls_pred = cls_pred[conf_inds, :]
                conf_pred = conf_pred[conf_inds]
                
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < conf_pred.size(0):
                _, topk_inds = torch.topk(conf_pred, nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_pred = cls_pred[topk_inds, :]
                conf_pred = conf_pred[topk_inds]

            # Save the result of current scale
            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)
        
        # Merge the results of different scales together
        multi_lvl_bboxes = torch.cat(multi_lvl_bboxes)
        multi_lvl_cls_scores = torch.cat(multi_lvl_cls_scores)
        multi_lvl_conf_scores = torch.cat(multi_lvl_conf_scores)

        if with_nms and (multi_lvl_conf_scores.size(0) == 0):
            return torch.zeros((0, 5)), torch.zeros((0, ))

        if rescale:
            if pad_param is not None:
                multi_lvl_bboxes -= multi_lvl_bboxes.new_tensor(
                    [pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
            multi_lvl_bboxes /= multi_lvl_bboxes.new_tensor(scale_factor)

        # the class_id for background is num_classes. i.e., the last column.
        padding = multi_lvl_cls_scores.new_zeros(multi_lvl_cls_scores.shape[0], 1)
        multi_lvl_cls_scores = torch.cat([multi_lvl_cls_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                multi_lvl_bboxes,
                multi_lvl_cls_scores,
                cfg['score_thr'],
                cfg['nms'],
                cfg['max_per_img'],
                score_factors=multi_lvl_conf_scores)
            return det_bboxes, det_labels
        else:
            return (multi_lvl_bboxes, multi_lvl_cls_scores, multi_lvl_conf_scores)

    @force_fp32(apply_to=('pred_maps', ))
    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        num_imgs = len(img_metas)
        device = pred_maps[0][0].device

        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        responsible_flag_list = []
        for img_id in range(len(img_metas)):
            responsible_flag_list.append(
                self.anchor_generator.responsible_flags(
                    featmap_sizes, gt_bboxes[img_id], device))

        target_maps_list, neg_maps_list = self.get_targets(
            anchor_list, responsible_flag_list, gt_bboxes, gt_labels, img_metas)

        losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(
            self.loss_single, pred_maps, target_maps_list, neg_maps_list)

        return dict(
            loss_cls=losses_cls,
            loss_conf=losses_conf,
            loss_xy=losses_xy,
            loss_wh=losses_wh)

    def loss_single(self, pred_map, target_map, neg_map):
        """Compute loss of a single image from a batch.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            target_map (Tensor): The Ground-Truth target for a single level.
            neg_map (Tensor): The negative masks for a single level.

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(num_imgs, -1, self.num_attrib)
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 4]
        pos_and_neg_mask = neg_mask + pos_mask
        pos_mask = pos_mask.unsqueeze(dim=-1)
        if torch.max(pos_and_neg_mask) > 1.:
            warnings.warn('There is overlap between pos and neg sample.')
            pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

        pred_xy = pred_map[..., :2]
        pred_wh = pred_map[..., 2:4]
        pred_conf = pred_map[..., 4]
        pred_label = pred_map[..., 5:]

        target_xy = target_map[..., :2]
        target_wh = target_map[..., 2:4]
        target_conf = target_map[..., 4]
        target_label = target_map[..., 5:]

        loss_cls = self.loss_cls(pred_label, target_label, weight=pos_mask)
        loss_conf = self.loss_conf(
            pred_conf, target_conf, weight=pos_and_neg_mask)
        loss_xy = self.loss_xy(pred_xy, target_xy, weight=pos_mask)
        loss_wh = self.loss_wh(pred_wh, target_wh, weight=pos_mask)

        return loss_cls, loss_conf, loss_xy, loss_wh

    def get_targets(self, anchor_list, responsible_flag_list, gt_bboxes_list,
                    gt_labels_list, img_metas):
        """Compute target maps for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi level responsible
                flags of each image. Each element is a tensor of shape
                (num_total_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """

        num_imgs = len(anchor_list)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        results = multi_apply(self._get_targets_single, anchor_list,
                              responsible_flag_list, gt_bboxes_list,
                              gt_labels_list)

        all_target_maps, all_neg_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)

        return target_maps_list, neg_maps_list

    def _get_targets_single(self, anchors, responsible_flags, gt_bboxes,
                            gt_labels):
        """Generate matching bounding box prior and converted GT.

        Args:
            anchors (list[Tensor]): Multi-level anchors of the image.
            responsible_flags (list[Tensor]): Multi-level responsible flags of
                anchors
            gt_bboxes (Tensor): Ground truth bboxes of single image.
            gt_labels (Tensor): Ground truth labels of single image.

        Returns:
            tuple:
                target_map (Tensor): Predication target map of each
                    scale level, shape (num_total_anchors,
                    5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """

        anchor_strides = []
        for i in range(len(anchors)):
            anchor_strides.append(
                torch.tensor(self.featmap_strides[i],
                             device=gt_bboxes.device).repeat(len(anchors[i])))
        concat_anchors = torch.cat(anchors)
        concat_responsible_flags = torch.cat(responsible_flags)

        anchor_strides = torch.cat(anchor_strides)
        assert len(anchor_strides) == len(concat_anchors) == \
               len(concat_responsible_flags)
        assign_result = self.assigner.assign(concat_anchors,
                                             concat_responsible_flags,
                                             gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, concat_anchors,
                                              gt_bboxes)

        target_map = concat_anchors.new_zeros(
            concat_anchors.size(0), self.num_attrib)

        target_map[sampling_result.pos_inds, :4] = self.bbox_coder.encode(
            sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes,
            anchor_strides[sampling_result.pos_inds])

        target_map[sampling_result.pos_inds, 4] = 1

        gt_labels_one_hot = F.one_hot(
            gt_labels, num_classes=self.num_classes).float()
        if self.one_hot_smoother != 0:  # label smooth
            gt_labels_one_hot = gt_labels_one_hot * (
                1 - self.one_hot_smoother
            ) + self.one_hot_smoother / self.num_classes
        target_map[sampling_result.pos_inds, 5:] = gt_labels_one_hot[
            sampling_result.pos_assigned_gt_inds]

        neg_map = concat_anchors.new_zeros(
            concat_anchors.size(0), dtype=torch.uint8)
        neg_map[sampling_result.neg_inds] = 1

        return target_map, neg_map

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
