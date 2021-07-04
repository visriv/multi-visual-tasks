from ..model_builder import DETECTORS
from .base_detectors import SingleStageDetector


@DETECTORS.register_module()
class ATSS(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self, cfg):
        super(ATSS, self).__init__(cfg)
