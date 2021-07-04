from ..model_builder import DETECTORS
from .base_detectors import SingleStageDetector


@DETECTORS.register_module()
class FOVEA(SingleStageDetector):
    """Implementation of `FoveaBox <https://arxiv.org/abs/1904.03797>`_"""

    def __init__(self, cfg):
        super(FOVEA, self).__init__(cfg)
