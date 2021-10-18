# -*- coding: utf-8 -*-
# @Time    : 2020/12/02 16:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com

import functools
import torch

dataset_aliases = {
    "voc": ["VOCDataset"],
    "coco": ["CocoDataset"],
    "retail_one_det": ["DetRetailOneDataset"],
    "kitti": ["KittiDataset"]
}


def assert_tensor_type(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f"{args[0].__class__.__name__} has no attribute "
                f"{func.__name__} for type {args[0].datatype}"
            )
        return func(*args, **kwargs)

    return wrapper


def replace_ImageToTensor(pipelines):
    """Replace the ImageToTensor transform in a data pipeline to
    DefaultFormatBundle, which is normally useful in batch inference.
    Args:
        pipelines (list[dict]): Data pipeline configs.
    Returns:
        list: The new pipeline list with all ImageToTensor replaced by
            DefaultFormatBundle.
    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='JointResize', keep_ratio=True),
        ...            dict(type='JointRandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='ImageToTensor', keys=['img']),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='JointResize', keep_ratio=True),
        ...            dict(type='JointRandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='DefaultFormatBundle'),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> assert expected_pipelines == replace_ImageToTensor(pipelines)
    """
    for key_p, value_p in pipelines.items():
        if key_p == "MultiScaleFlipAug":
            assert "transforms" in value_p
            replace_ImageToTensor(value_p.transforms)
        elif key_p == "ImageToTensor":
            pipelines[key_p] = "DefaultFormatBundle"
    return pipelines


def voc_classes():
    return [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]


def coco_classes():
    return [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic_light",
        "fire_hydrant",
        "stop_sign",
        "parking_meter",
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
        "sports_ball",
        "kite",
        "baseball_bat",
        "baseball_glove",
        "skateboard",
        "surfboard",
        "tennis_racket",
        "bottle",
        "wine_glass",
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
        "hot_dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted_plant",
        "bed",
        "dining_table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell_phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy_bear",
        "hair_drier",
        "toothbrush",
    ]


def kitti_classes():
    return [
        'car',
        'pedestrian',
        'cyclist'
    ]


def retail_one_det_classes():
    return ["retail"]


def get_classes(dataset_class):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if isinstance(dataset_class, str):
        if dataset_class in alias2name:
            labels = eval(alias2name[dataset_class] + "_classes()")
        else:
            raise ValueError(f"Unrecognized dataset: {dataset_class}")
    else:
        raise TypeError(f"dataset must a str, but got {type(dataset_class)}")
    return labels
