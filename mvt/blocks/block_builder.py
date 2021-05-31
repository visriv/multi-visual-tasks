# -*- coding: utf-8 -*-
# @Time    : 2020/12/02 16:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com
# @File    : block_builder.py

import inspect
from torch import nn
from yacs.config import CfgNode

from mvt.utils.reg_util import (Registry, build_module_from_cfg, 
                                build_module_from_dict)
from mvt.utils.config_util import convert_to_dict

# --------------------------------------------------------------------------- #
# Registries for blocks
# --------------------------------------------------------------------------- #
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
PIXEL_SAMPLERS = Registry('pixel sampler')


def build_block(cfg, registry, default_args=None):
    """Build a module.
    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_module_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    elif isinstance(cfg, CfgNode): # CfgNode is inheritted from dict
        return build_module_from_cfg(cfg, registry, default_args)
    else:
        return build_module_from_dict(cfg, registry, default_args) 


def build_backbone(cfg):
    """Build backbone."""
    return build_block(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build_block(cfg, NECKS)


def build_roi_extractor(cfg, default_args=None):
    """Build roi extractor."""
    return build_block(cfg, ROI_EXTRACTORS, default_args)


def build_shared_head(cfg, default_args=None):
    """Build shared head."""
    return build_block(cfg, SHARED_HEADS, default_args)


def build_head(cfg, default_args=None):
    """Build head."""
    return build_block(cfg, HEADS, default_args)


def build_loss(cfg, default_args=None):
    """Build loss."""
    return build_block(cfg, LOSSES, default_args)


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    return build_module_from_cfg(cfg, PIXEL_SAMPLERS, default_args)
