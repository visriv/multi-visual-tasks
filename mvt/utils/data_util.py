# -*- coding: utf-8 -*-
# @Time    : 2020/12/02 16:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com
# @File    : dataset_util.py
import warnings
import functools
import torch

dataset_aliases = {
    'voc': ['VOCDataset'],
    'coco': ['CocoDataset'],
    'cityscapes': ['CityscapesDataset'],
    'voc_seg': ['SegVOCDataset'],
    'animal_det': ['DetAnimalDataset'],
    'cdhead_det': ['CatDogHeadDataset'],
    'multi_objdet': ['MultiObjectDataset']
}


def assert_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f'{args[0].__class__.__name__} has no attribute '
                f'{func.__name__} for type {args[0].datatype}')
        return func(*args, **kwargs)

    return wrapper


class DataContainer:
    """A container for any type of objects.
    
    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    `DataContainer` to overcome these limitations. 
    The behavior can be either of the following.
    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """
    
    def __init__(self,
                 data,
                 stack=False,
                 padding_value=0,
                 cpu_only=False,
                 pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.data)})'

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()


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
        if key_p == 'MultiScaleFlipAug':
            assert 'transforms' in value_p
            replace_ImageToTensor(value_p.transforms)
        elif key_p == 'ImageToTensor':
            pipelines[key_p] = 'DefaultFormatBundle'
    return pipelines


def voc_classes():
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]


def coco_classes():
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]


def animal_det_classes():
    return [
        'golden-retriever', 'husky', 'teddy-dog', 'shiba-inu', 'samoyed',
        'corgi', 'alaskan', 'labrador', 'hyena', 'german-shepherd',
        'pomeranian', 'french-bulldog', 'bichon-frise', 'horse-dog',
        'tibetan-mastiff', 'border-collie', 'pug', 'chow-chow', 
        'pit-bull', 'chihuahua', 'rottweiler', 'doberman', 'bull-terrier', 
        'english-bulldog', 'akita', 'dachshund', 'caucasian-dog', 'bully', 
        'giant-poodle', 'great-dane', 'schnauzer', 'teacup-dogs', 'shar-pei', 
        'beagles', 'boxer', 'orange-cat', 'lynx', 'doll-cat', 
        'british-shorthair-cat', 'folds', 'hairless-cat', 'caracal-cat', 
        'tabby-cat', 'siamese-cat', 'persian-cat', 'american-shorthair', 
        'maine-coon', 'russian-blue-cat', 'cow-cat', 'bengal-cat', 
        'squirrel', 'fox', 'kangaroo', 'deer', 'zebra', 'wildebeest', 
        'honey-badger', 'rhinoceros', 'giraffe', 'african-wild-dog', 'seal', 
        'hedgehog', 'groundhog', 'donkey', 'brown-bear', 'black-bear',
        'polar-bear', 'grizzly-bear', 'raccoon', 'dinosaur', 'baboon', 
        'warthog', 'camel', 'alpaca', 'koala', 'weasel', 'sloth', 'snow-leopard', 
        'elk', 'rabbit', 'lion', 'giant-panda', 'tiger', 'monkey', 'elephant', 
        'hamster', 'pig', 'cattle', 'sheep', 'horse', 'orangutan', 'antelope', 
        'cheetah', 'wild-boar', 'fish', 'chickens', 'goose', 'penguin', 'hippo', 
        'parrot', 'mouse', 'cathrow', 'longhair-cat', 'mini-doberman', 'garfield', 
        'silver-gradient', 'short-footed-cat', 'otter', 'shetland', 'sika-deer'
    ]


def cdhead_det_classes():
    return ['cat', 'dog']


def multi_objdet_classes():
    return [
        'person', 'cartoon-person', 'game-role', 'cat', 'dog', 'snake',
        'bird', 'fish', 'rabbit', 'monkey', 'horse', 'chicken', 'pig',
        'cow', 'sheep', 'bicycle', 'tricycle', 'motorbike', 'tractor',
        'car', 'bus', 'truck', 'excavator', 'crane', 'train', 'plane',
        'tank', 'ship', 'villa', 'pavilion', 'tower', 'temple', 'palace', 
        'chair', 'bed', 'table', 'sofa', 'bench', 'vase', 'potted-plant', 
        'bag', 'umbrella', 'computer', 'television', 'lamp', 'mouse', 
        'keyboard', 'cell-phone', 'dish', 'bowl', 'spoon', 'bottle', 'cup', 
        'fork', 'pot', 'knife', 'basketball', 'skateboard', 'book', 'banana', 
        'apple', 'orange', 'watermelon', 'pizza', 'cake'
    ]


def voc_seg_classes():
    return [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]


def voc_seg_palette():
    """Pascal VOC palette for external use."""
    return [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
            [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
            [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def cityscapes_classes():
    """Cityscapes class names for external use."""
    return [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


def cityscapes_palette():
    """Cityscapes palette for external use."""
    return [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32]]


def get_classes(dataset_class):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if isinstance(dataset_class, str):
        if dataset_class in alias2name:
            labels = eval(alias2name[dataset_class] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset_class}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset_class)}')
    return labels
