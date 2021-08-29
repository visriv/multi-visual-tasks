import numpy as np
import torch
from yacs.config import CfgNode

from mvt.utils.parallel_util import scatter
from mvt.utils.runtime_util import collate
from mvt.utils.io_util import imread
from mvt.utils.checkpoint_util import load_checkpoint
from mvt.utils.data_util import get_classes
from mvt.datasets.transforms import Compose
from mvt.models.model_builder import build_model


class LoadImageToDetector(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results["img"], str):
            results["filename"] = results["img"]
            results["ori_filename"] = results["img"]
        else:
            results["filename"] = None
            results["ori_filename"] = None
        img = imread(results["img"])
        results["img"] = img
        results["img_fields"] = ["img"]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        return results


def get_pipeline_list(pipeline_cfg, is_byte=False):
    """Get the list of configures for constructing pipelines

    Note:
        self.pipeline is a CfgNode

    Returns:
        list[dict]: list of dicts with types and parameters for constructing pipelines.
    """

    pipeline_list = []
    for k_t, v_t in pipeline_cfg.items():
        pipeline_item = {}
        if len(v_t) > 0:
            if not isinstance(v_t, CfgNode):
                raise TypeError("pipeline items must be a CfgNode")
        if is_byte and k_t == "LoadImageFromFile":
            # remove the load image process
            k_t = "LoadImageFromWebcam"
        pipeline_item["type"] = k_t

        for k_a, v_a in v_t.items():
            if isinstance(v_a, CfgNode):
                pipeline_item[k_a] = []
                for sub_kt, sub_vt in v_a.items():
                    sub_item = {}
                    if len(sub_vt) > 0:
                        if not isinstance(sub_vt, CfgNode):
                            raise TypeError("transform items must be a CfgNode")
                    sub_item["type"] = sub_kt
                    for sub_ka, sub_va in sub_vt.items():
                        if isinstance(sub_va, CfgNode):
                            raise TypeError("Only support two built-in layers")
                        sub_item[sub_ka] = sub_va
                    pipeline_item[k_a].append(sub_item)
            else:
                pipeline_item[k_a] = v_a
        pipeline_list.append(pipeline_item)

    return pipeline_list


def get_detector(config, checkpoint=None, device="cuda:0"):
    """Initialize a detector from config file.

    Args:
        config (CfgNode): cfg
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """

    config.MODEL.PRETRAINED_MODEL_PATH = ""
    model = build_model(config.MODEL)
    if checkpoint is not None:
        map_loc = "cpu" if device == "cpu" else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if "CLASSES" in checkpoint["meta"]:
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = get_classes(config.DATA.NAME)

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model


def inference_detector(config, model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """

    device = next(model.parameters()).device  # model device
    # prepare data
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
        # set loading pipeline type
        pipeline_list = get_pipeline_list(config.DATA.TEST_TRANSFORMS, is_byte=True)
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)
        pipeline_list = get_pipeline_list(config.DATA.TEST_TRANSFORMS)

    # build the data pipeline
    test_pipeline = Compose(pipeline_list)
    data = test_pipeline(data)
    data = collate([data])

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # just get the actual data
        data["img_metas"] = data["img_metas"][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)[0]
    return result


def show_detector_result(
    model, img, result, score_thr=0.3, with_show=True, save_path=None
):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, "module"):
        model = model.module
    img = model.show_result(
        img, result, score_thr=score_thr, show=with_show, out_file=save_path
    )
