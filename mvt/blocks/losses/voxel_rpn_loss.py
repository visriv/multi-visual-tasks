from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn

from mvt.cores.bbox.bbox_transforms import limit_period_torch
from ..block_builder import LOSSES


def indices_to_dense_vector(indices, size, indices_value=1., default_value=0):
    """
    Creates dense vector with indices set to specific value and rest to zeros. This function exists because
    it is unclear if it is safe to use tf.sparse_to_dense(indices, [size], 1, validate_indices=False) with
    indices which are not ordered. This function accepts a dynamic size (e.g. tf.shape(tensor)[0])
    :param indices: 1d Tensor with integer indices which are to be set to indices_values.
    :param size: scalar with size (integer) of output Tensor.
    :param indices_value: values of elements specified by indices in the output vector
    :param default_value: values of other elements in the output vector.
    :return: dense 1D Tensor of shape [size] with indices set to indices_values and the rest set to default_value.
    """
    dense = torch.zeros(size).fill_(default_value)
    dense[indices] = indices_value

    return dense


def _sigmoid_cross_entropy_with_logits(logits, labels):
    """
    get the sigmoid cross entropy by calculating with logits and labels
    :param logits: predicted logits
    :param labels: ground truth labels
    :return: loss
    """
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    return loss


def _softmax_cross_entropy_with_logits(logits, labels):
    """
    get the softmax cross entropy by calculating with logits and labels
    :param logits: predicted logits
    :param labels: ground truth labels
    :return: loss
    """
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param) # [N, ..., C] -> [N, C, ...]
    loss_ftor = nn.CrossEntropyLoss(reduction='none')
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss


def one_hot(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    """get the one hot tensor from tensor"""
    tensor_onehot = torch.zeros(*list(tensor.shape), depth, dtype=dtype, device=tensor.device)
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot


def add_sin_difference(boxes1, boxes2, boxes1_rot, boxes2_rot):
    """
    add sin function to rotation
    :param boxes1: predicted boxes
    :param boxes2: ground truth boxes
    :param boxes1_rot: rotation of predicted boxes
    :param boxes2_rot: rotation of ground truth boxes
    :param factor: the scaling factor
    :return: boxes with rotation encoded by sin
    """
    rad_pred_encoding = torch.sin(boxes1_rot) * torch.cos(boxes2_rot)
    rad_tg_encoding = torch.cos(boxes1_rot) * torch.sin(boxes2_rot)
    boxes1 = torch.cat([boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
    boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]], dim=-1)
    return boxes1, boxes2


def get_direction_target(anchors, reg_targets, is_one_hot=True, dir_offset=0, num_bins=2):
    """
    get the directions of targets
    :param anchors: given anchors
    :param reg_targets: selected targets
    :param is_one_hot: whether using one hot prediction
    :param dir_offset: the offset of direction
    :param num_bins: the number of direction bins for classification
    :return: the selected ground truth directions for targets
    """
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = limit_period_torch(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if is_one_hot:
        dir_cls_targets = one_hot(dir_cls_targets, num_bins, dtype=anchors.dtype)
    return dir_cls_targets


class Loss(object):
    """Abstract base class for loss functions."""
    __metaclass__ = ABCMeta

    def __call__(self, prediction_tensor, target_tensor, ignore_nan_targets=False, scope=None, **params):
        """
        Call the loss function.
        :param prediction_tensor: an N-d tensor of shape [batch, anchors, ...] representing predicted quantities.
        :param target_tensor: an N-d tensor of shape [batch, anchors, ...] representing regression
               or classification targets.
        :param ignore_nan_targets: whether to ignore nan targets in the loss computation.
               E.g. can be used if the target tensor is missing groundtruth data that
               shouldn't be factored into the loss.
        :param scope: Op scope name. Defaults to 'Loss' if None.
        :param params: Additional keyword arguments for specific implementations of the Loss.
        :return: loss, a tensor representing the value of the loss function.
        """
        if ignore_nan_targets:
            target_tensor = torch.where(torch.isnan(target_tensor), prediction_tensor, target_tensor)
        return self._compute_loss(prediction_tensor, target_tensor, **params)

    @abstractmethod
    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        """
        Method to be overridden by implementations.
        :param prediction_tensor: a tensor representing predicted quantities
        :param target_tensor: a tensor representing regression or classification targets
        :param params: Additional keyword arguments for specific implementations of the Loss.
        :return: loss, an N-d tensor of shape [batch, anchors, ...] containing the loss per anchor
        """
        pass


class SigmoidFocalClassificationLoss(Loss):
    """
    Sigmoid focal cross entropy loss. Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        """
        Constructor.
        :param gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        :param alpha: optional alpha weighting factor to balance positives vs negatives.
        """
        self._alpha = alpha
        self._gamma = gamma

    def _compute_loss(self, prediction_tensor, target_tensor, weights=None, class_indices=None):
        """
        Compute loss function.
        :param prediction_tensor: A float tensor of shape [batch_size, num_anchors, num_classes]
               representing the predicted logits for each class
        :param target_tensor: A float tensor of shape [batch_size, num_anchors,
               num_classes] representing one-hot encoded classification targets
        :param weights: a float tensor of shape [batch_size, num_anchors]
        :param class_indices: (Optional) A 1-D integer tensor of class indices.
               If provided, computes loss only for the specified class indices.
        :return: loss, a float tensor of shape [batch_size, num_anchors, num_classes]
                 representing the value of the loss function.
        """
        if weights is None:
            raise Exception("weights should not be none for SigmoidFocalClassificationLoss")

        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(labels=target_tensor, logits=prediction_tensor))
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) + ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
        if class_indices is not None:
            weights = weights.unsqueeze(2)
            weights *= indices_to_dense_vector(
                class_indices, prediction_tensor.shape[2]).view(1, 1, -1).type_as(prediction_tensor)
            return focal_cross_entropy_loss * weights
        else:
            weights = weights.unsqueeze(2)
            return focal_cross_entropy_loss * weights


class WeightedSoftmaxClassificationLoss(Loss):
    """Softmax loss function."""
    def __init__(self, logit_scale=1.0):
        """
        Constructor.
        :param logit_scale: When this value is high, the prediction is "diffused" and
               when this value is low, the prediction is made peakier. (default 1.0)
        """
        self._logit_scale = logit_scale

    def _compute_loss(self, prediction_tensor, target_tensor, weights=None):
        """
        Compute loss function.
        :param prediction_tensor: A float tensor of shape [batch_size, num_anchors, num_classes]
               representing the predicted logits for each class
        :param target_tensor: A float tensor of shape [batch_size, num_anchors, num_classes]
               representing one-hot encoded classification targets
        :param weights: a float tensor of shape [batch_size, num_anchors]
        :return: loss, a float tensor of shape [batch_size, num_anchors] representing the value of the loss function.
        """
        if weights is None:
            raise Exception("weights should not be none for WeightedSoftmaxClassificationLoss")
        num_classes = prediction_tensor.shape[-1]
        prediction_tensor = torch.div(prediction_tensor, self._logit_scale)
        per_row_cross_ent = (_softmax_cross_entropy_with_logits(
            labels=target_tensor.view(-1, num_classes), logits=prediction_tensor.view(-1, num_classes)))
        return per_row_cross_ent.view(weights.shape) * weights


class WeightedSmoothL1LocalizationLoss(Loss):
    """
    Smooth L1 localization loss function.
    The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
    otherwise, where x is the difference between predictions and target.
    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """
    def __init__(self, sigma=3.0):
        """
        Constructor
        :param sigma: parameter for computing Smooth L1 loss
        """
        super().__init__()
        self._sigma = sigma

    def _compute_loss(self, prediction_tensor, target_tensor, weights=None):
        """
        Compute loss function.
        :param prediction_tensor: A float tensor of shape [batch_size, num_anchors,
               code_size] representing the (encoded) predicted locations of objects.
        :param target_tensor: A float tensor of shape [batch_size, num_anchors, code_size]
               representing the regression targets
        :param weights: a float tensor of shape [batch_size, num_anchors]
        :return: loss: a float tensor of shape [batch_size, num_anchors] tensor
                 representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor

        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma**2)).type_as(abs_diff)
        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) \
            + (abs_diff - 0.5 / (self._sigma**2)) * (1. - abs_diff_lt_1)

        anchorwise_smooth_l1norm = loss
        if weights is not None:
            anchorwise_smooth_l1norm *= weights.unsqueeze(-1)

        return anchorwise_smooth_l1norm


class SigmoidConfidenceLoss(Loss):
    """
    Sigmoid confidence cross entropy loss.
    """
    def __init__(self):
        super().__init__()

    def _compute_loss(self, prediction_tensor, target_tensor, weights=None):
        """
        Compute loss function.
        :param prediction_tensor: A float tensor of shape [batch_size, num_obj]
               representing the predicted logits for each object
        :param target_tensor: A float tensor of shape [batch_size, num_obj]
        :param weights: a float tensor of shape [batch_size, num_obj]
        :return: loss, a float tensor of shape [batch_size, num_anchors, num_classes]
                 representing the value of the loss function.
        """
        prediction_probabilities = torch.sigmoid(prediction_tensor)

        loss = -target_tensor*torch.log(prediction_probabilities) - \
               (1. - target_tensor) * torch.log(1. - prediction_probabilities)

        if weights is None:
            return loss
        else:
            return loss * weights


@LOSSES.register_module()
class VoxelRPNLoss(nn.Module):
    def __init__(
        self,
        classification_loss=None,
        localization_loss=None,
        cls_weight=1.0,
        loc_weight=2.0,
        dir_weight=0.2,
        box_coder_size=7,
        num_classes=3
    ):
        super(VoxelRPNLoss, self).__init__()
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.dir_weight = dir_weight
        self.num_classes = num_classes
        self.box_coder_size = box_coder_size
        cls_loss_type = classification_loss["type"]
        if cls_loss_type == 'weighted_sigmoid_focal':
            self.cls_loss_ftor = SigmoidFocalClassificationLoss(
                gamma=classification_loss["gamma"],
                alpha=classification_loss["alpha"])
        else:
            raise Exception("Class loss type error!")

        loc_loss_type = localization_loss["type"]
        if loc_loss_type == 'weighted_smooth_l1':
            self.loc_loss_ftor = WeightedSmoothL1LocalizationLoss(
                localization_loss["sigma"])
        else:
            raise Exception("Localization loss type error!")

    def forward(
        self,
        cls_preds,
        box_preds,
        dir_preds,
        cls_targets,
        reg_targets,
        cls_weights,
        reg_weights,
        labels,
        anchors,
        **kwargs
    ):
        batch_size = int(box_preds.shape[0])
        box_preds = box_preds.view(batch_size, -1, self.box_coder_size)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)

        cls_targets = cls_targets.squeeze(-1)
        one_hot_targets = one_hot(cls_targets, depth=self.num_class + 1, dtype=box_preds.dtype)

        one_hot_targets = one_hot_targets[..., 1:]

        box_preds, reg_targets = add_sin_difference(
                box_preds, reg_targets, box_preds[..., 6:7], reg_targets[..., 6:7])

        loc_losses = self.loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)  # [N, M]
        cls_losses = self.cls_loss_ftor(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]

        loc_loss_reduced = loc_losses.sum() / batch_size
        loc_loss_reduced *= self.loc_weight

        cls_loss_reduced = cls_losses.sum() / batch_size
        cls_loss_reduced *= self.cls_weight

        dir_targets = get_direction_target(anchors, reg_targets)
        weights = (labels > 0).type_as(dir_preds)
        weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
        dir_loss = self._dir_loss_ftor(dir_preds, dir_targets, weights=weights)
        dir_loss_reduced = dir_loss.sum() / batch_size * self.dir_weight
        
        return loc_loss_reduced, cls_loss_reduced, dir_loss_reduced
