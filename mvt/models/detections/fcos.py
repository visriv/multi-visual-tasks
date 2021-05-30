from ..model_builder import DETECTORS
from .base_detectors import SingleStageDetector


@DETECTORS.register_module()
class FCOS(SingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self, cfg):
        super(FCOS, self).__init__(cfg)
