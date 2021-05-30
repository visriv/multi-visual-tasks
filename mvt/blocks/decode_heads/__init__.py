from .base_decode_head import BaseDecodeHead
from .cascade_decode_head import BaseCascadeDecodeHead
from .fcn_head import SegFCNHead
from .fpn_head import SegFPNHead
from .ocr_head import OCRHead
from .fcn_map_head import SegFCNMapHead


__all__ = [
    'BaseDecodeHead', 'BaseCascadeDecodeHead', 'SegFCNHead',
    'SegFCNMapHead', 'SegFPNHead', 'OCRHead'
]
