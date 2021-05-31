from .fcn_head import SegFCNHead
from ..block_builder import HEADS
from mvt.utils.fp16_util import force_fp32
from mvt.cores.layer_ops import resize


@HEADS.register_module()
class SegFCNMapHead(SegFCNHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        loss['loss_seg'] = self.loss_decode(seg_logit, seg_label)
        return loss
    