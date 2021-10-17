from ..model_builder import DETECTORS
from .base_detectors import SingleStageDetector


@DETECTORS.register_module()
class YOLOV3(SingleStageDetector):
    def __init__(self, cfg):
        super(YOLOV3, self).__init__(cfg)


@DETECTORS.register_module()
class YOLOV4(SingleStageDetector):
    def __init__(self, cfg):
        super(YOLOV4, self).__init__(cfg)


@DETECTORS.register_module()
class YOLOV5(SingleStageDetector):
    def __init__(self, cfg):
        super(YOLOV5, self).__init__(cfg)
