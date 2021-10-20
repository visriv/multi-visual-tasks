from abc import ABCMeta
from abc import abstractmethod
from .bbox3d_util import (
    rbbox2d_to_near_bbox,
    iou_jit,
    distance_similarity
)


# Region Similarity
class RegionSimilarityCalculator(object):
    """Abstract base class for 2d region similarity calculator."""
    __metaclass__ = ABCMeta

    def compare(self, boxes1, boxes2):
        """
        Computes matrix of pairwise similarity between BoxLists.
        This op (to be overriden) computes a measure of pairwise similarity between
        the boxes in the given BoxLists. Higher values indicate more similarity.
        Note that this method simply measures similarity and does not explicitly perform a matching.
        :param boxes1: [N, 5] [x,y,w,l,r] tensor.
        :param boxes2: [M, 5] [x,y,w,l,r] tensor.
        :return: a (float32) tensor of shape [N, M] with pairwise similarity score.
        """
        return self._compare(boxes1, boxes2)

    @abstractmethod
    def _compare(self, boxes1, boxes2):
        pass


class NearestIouSimilarity(RegionSimilarityCalculator):
    """
    Class to compute similarity based on the squared distance metric.
    This class computes pairwise similarity between two BoxLists based on the negative squared distance metric.
    """
    def _compare(self, boxes1, boxes2):
        """
        Compute matrix of (negated) sq distances.
        :param boxes1: BoxList holding N boxes.
        :param boxes2: BoxList holding M boxes.
        :return: A tensor with shape [N, M] representing negated pairwise squared distance.
        """
        boxes1_bv = rbbox2d_to_near_bbox(boxes1)
        boxes2_bv = rbbox2d_to_near_bbox(boxes2)
        ret = iou_jit(boxes1_bv, boxes2_bv, eps=0.0)
        return ret


class DistanceSimilarity(RegionSimilarityCalculator):
    """
    Class to compute similarity based on Intersection over Area (IOA) metric.
    This class computes pairwise similarity between two BoxLists based on their
    pairwise intersections divided by the areas of second BoxLists.
    """
    def __init__(self, distance_norm, with_rotation=False, rotation_alpha=0.5):
        self._distance_norm = distance_norm
        self._with_rotation = with_rotation
        self._rotation_alpha = rotation_alpha

    def _compare(self, boxes1, boxes2):
        """
        Compute matrix of (negated) sq distances.
        :param boxes1: BoxList holding N boxes.
        :param boxes2: BoxList holding M boxes.
        :return: A tensor with shape [N, M] representing negated pairwise squared distance.
        """
        return distance_similarity(
            boxes1[..., [0, 1, -1]],
            boxes2[..., [0, 1, -1]],
            dist_norm=self._distance_norm,
            with_rotation=self._with_rotation,
            rot_alpha=self._rotation_alpha)
