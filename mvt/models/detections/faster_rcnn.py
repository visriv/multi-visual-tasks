from ..model_builder import DETECTORS
from .base_detectors import TwoStageDetector


@DETECTORS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, cfg):
        super(FasterRCNN, self).__init__(cfg)
