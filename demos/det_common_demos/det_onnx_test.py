import argparse
import numpy as np
import onnx
import onnxruntime as rt
import torch
import os
import os.path as osp

from mvt.utils.io_util import imread
from mvt.utils.geometric_util import imresize
from mvt.utils.photometric_util import imnormalize
from mvt.cores.ops import multiclass_nms
from mvt.utils.vis_util import imshow_det_bboxes


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class_names = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def preprocess_example_input(input_config):
    """Prepare an example input image for ``generate_inputs_and_wrap_model``.

    Args:
        input_config (dict): customized config describing the example input.

    Returns:
        tuple: (one_img, one_meta), tensor of the example input image and \
            meta information for the example input image.
    """

    input_path = input_config["input_path"]
    input_shape = input_config["input_shape"]
    one_img = imread(input_path)
    if "normalize_cfg" in input_config.keys():
        normalize_cfg = input_config["normalize_cfg"]
        mean = np.array(normalize_cfg["mean"], dtype=np.float32)
        std = np.array(normalize_cfg["std"], dtype=np.float32)
        one_img = imnormalize(one_img, mean, std)

    # ratio of width/height for the original picture
    ori_shape = one_img.shape
    new_shape = input_shape
    print("ori_shape：", ori_shape)
    print("new_shape：", new_shape)

    one_img = imresize(one_img, input_shape[2:][::-1]).transpose(2, 0, 1)

    print("one_img imresize success!!!")
    print(one_img.shape)

    one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)

    print("one_img torch from_numpy success!!!")

    _, C, H, W = input_shape
    one_meta = {
        "img_shape": (H, W, C),
        "ori_shape": ori_shape,
        "pad_shape": (H, W, C),
        "filename": "<demo>.png",
        "scale_factor": 1.0,
        "flip": False,
    }

    return one_img, one_meta


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MVT models to ONNX")
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument("--input-img", type=str, help="Images for input")
    # parser.add_argument('--show', action='store_true', help='show onnx graph')
    # parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument("--opset-version", type=int, default=11)
    # parser.add_argument(
    #     '--verify',
    #     action='store_true',
    #     help='verify the onnx model output against pytorch output')
    parser.add_argument(
        "--shape", type=int, nargs="+", default=[416, 416], help="input image size"
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=[0, 0, 0],
        help="mean value used for preprocess input data",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=[255, 255, 255],
        help="variance value used for preprocess input data",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # opset version
    assert args.opset_version == 11, "Only support opset 11 now"
    # input img
    if not args.input_img:
        args.input_img = osp.join(
            osp.dirname(__file__), "../meta/test_data/a0519qvbyom_001.jpg"
        )
    # input shape
    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")
    # normalize_cfg
    assert len(args.mean) == 3
    assert len(args.std) == 3
    normalize_cfg = {"mean": args.mean, "std": args.std}
    # onnx model
    output_file = "meta/onnx_models/det_yolov4_cspdarknet_coco.onnx"

    # preprocessing
    input_config = {
        "input_shape": input_shape,
        "input_path": args.input_img,
        "normalize_cfg": normalize_cfg,
    }
    one_img, one_meta = preprocess_example_input(input_config)

    # print("one_meta:", one_meta)

    # load onnx
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)

    # check the numerical value
    # get pytorch output
    # pytorch_result = model(tensor_data, [[one_meta]], return_loss=False)

    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1
    sess = rt.InferenceSession(output_file)

    # from mvt.cores.bbox import bbox2result
    out_list = sess.run(None, {net_feed_input[0]: one_img.detach().numpy()})

    ml_bboxes = out_list[0]
    ml_cls_scores = out_list[1]
    ml_conf_scores = out_list[2]

    # # only compare a part of result
    conf_thr = 0.005
    conf_inds = np.where(ml_conf_scores > conf_thr)
    ml_bboxes = ml_bboxes[conf_inds]
    ml_cls_scores = ml_cls_scores[conf_inds]
    ml_conf_scores = ml_conf_scores[conf_inds]

    nms_cfg = {"type": "common_nms", "iou_threshold": 0.45}

    det_bboxes, det_labels = multiclass_nms(
        torch.from_numpy(ml_bboxes),
        torch.from_numpy(ml_cls_scores),
        0.05,
        nms_cfg,
        100,
        score_factors=ml_conf_scores,
    )
    # only compare a part of result
    bbox_results = bbox2result(det_bboxes, det_labels, len(class_names))
    # print('bbox_results:', bbox_results)

    bboxes = np.vstack(bbox_results)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_results)
    ]
    labels = np.concatenate(labels)

    score_thr = 0.3
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    h_scale = one_meta["ori_shape"][0] / one_meta["img_shape"][0]
    w_scale = one_meta["ori_shape"][1] / one_meta["img_shape"][1]

    bboxes[:, 0] = bboxes[:, 0] * w_scale
    bboxes[:, 1] = bboxes[:, 1] * h_scale
    bboxes[:, 2] = bboxes[:, 2] * w_scale
    bboxes[:, 3] = bboxes[:, 3] * h_scale

    print("bboxes:", bboxes)
    print("labels:", labels)

    imshow_det_bboxes(args.input_img, bboxes, labels, class_names=class_names)
