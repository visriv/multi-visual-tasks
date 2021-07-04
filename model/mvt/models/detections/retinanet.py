from ..model_builder import DETECTORS
from .base_detectors import SingleStageDetector


@DETECTORS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self, cfg):
        super(RetinaNet, self).__init__(cfg)
