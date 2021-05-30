from ..model_builder import DETECTORS
from .base_detectors import SingleStageDetector


@DETECTORS.register_module()
class SSD(SingleStageDetector):
    
    def __init__(self, cfg):
        super(SSD, self).__init__(cfg)
