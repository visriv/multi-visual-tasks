from ..model_builder import DETECTORS
from .base_detectors import TwoStageDetector


@DETECTORS.register_module()
class MaskRCNN(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self, cfg):
        super(MaskRCNN, self).__init__(cfg)
