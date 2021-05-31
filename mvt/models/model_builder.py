# -*- coding: utf-8 -*-
# @Time    : 2020/12/02 22:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com
# @File    : block_builder.py

from mvt.utils.reg_util import Registry, build_model_from_cfg


# --------------------------------------------------------------------------- #
# Registries for blocks
# --------------------------------------------------------------------------- #
CLASSIFIERS = Registry("classifier")
DETECTORS = Registry("detector")
SEGMENTORS = Registry("segmentor")
REGRESSORS = Registry("regressor")
POSERS = Registry("poser")
MULTITASKERS = Registry("multitasker")


def build_model(cfg):
    """Build model."""
    if "TYPE" not in cfg:
        raise KeyError("cfg must have key \'TYPE\' to define model type")
    
    model_type = cfg.TYPE

    if model_type == "cls":
        return build_model_from_cfg(cfg, CLASSIFIERS)
    elif model_type == "det":
        return build_model_from_cfg(cfg, DETECTORS)
    elif model_type == "seg":
        return build_model_from_cfg(cfg, SEGMENTORS)
    elif model_type == "reg":
        return build_model_from_cfg(cfg, REGRESSORS)
    elif model_type == "pos":
        return build_model_from_cfg(cfg, POSERS)
    else:
        raise TypeError(f"No type for the task {model_type}")
