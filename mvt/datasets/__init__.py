# -*- coding: utf-8 -*-
# @Time    : 2021/1/7 17:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com

#from .classifications import *
from .detections import *
#from .segmentations import *
#from .regressions import *
#from .pose_estimations import *
from .transforms import *
from .data_sampler import (DistributedSampler, GroupSampler, 
                           DistributedGroupSampler)
from .data_builder import build_dataset, build_dataloader
from .data_wrapper import (DATASETS, PIPELINES, ConcatDataset, 
                           RepeatDataset, ClassBalancedDataset)


__all__ = [
    'DetBaseDataset', 'VOCDataset', 'CocoDataset', 'MosaicDataset',
    'DetAnimalDataset', 'CatDogHeadDataset', 'MultiObjectDataset',
    'Compose', 'to_tensor', 'ToTensor', 'ImageToTensor',
    'Transpose', 'Collect', 'ClsCollect', 'RegCollect', 'PosCollect', 
    'DefaultFormatBundle', 'LoadAnnotations',
    'LoadImageFromFile', 'LoadImageFromWebcam',
    'LoadMultiChannelImageFromFiles', 'LoadProposals', 'MultiScaleFlipAug',
    'JointResize', 'JointRandomFlip', 'Pad', 'JointRandomCrop', 'Normalize', 'SegRescale',
    'MinIoURandomCrop', 'Expand', 'PhotoMetricDistortion', 'Albu',
    'InstaBoost', 'RandomCenterCropPad', 'AutoAugment', 'CutOut', 'Shear',
    'Rotate', 'ColorTransform', 'EqualizeTransform', 'BrightnessTransform',
    'ContrastTransform', 'Translate', 'SegRandomCrop',  'GenerateHeatMap',
    'RandomGrayscale', 'ImgResize', 'ImgRandomFlip', 'ImgCenterCrop',
    'ImgRandomCrop', 'ImgRandomResizedCrop',
    'build_dataset', 'build_dataloader',
    'DATASETS', 'PIPELINES', 'ConcatDataset', 'RepeatDataset', 
    'ClassBalancedDataset', 'BboxCluesDataset',
    'DistributedSampler', 'GroupSampler', 'DistributedGroupSampler',
    'CIFAR10', 'CIFAR100', 'ClsBaseDataset', 'ImageNet', 'MNIST', 'FashionMNIST',
    'ClarityDataset', 'BeautyDataset', 'RotationDataset', 'BareDegreeDataset',
    'CoverQualityDataset', 'ZhongTaiCommonDataset',
    'SegBaseDataset', 'SegVOCDataset', 'CityscapesDataset', 'SegConcatDataset',
    'SegHeatMapConcatDataset',
    'PosMNodesTopDwonDataset'
    ]
