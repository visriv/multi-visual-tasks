# -*- coding: utf-8 -*-
# @Time    : 2020/12/02 22:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com

from model.mvt.utils.reg_util import Registry, build_model_from_cfg


# --------------------------------------------------------------------------- #
# Registries for blocks
# --------------------------------------------------------------------------- #
CLASSIFIERS = Registry("classifier")
DETECTORS = Registry("detector")
EMBEDDERS = Registry("embedder")


def build_model(cfg):
    """Build model."""
    if "TYPE" not in cfg:
        raise KeyError("cfg must have key 'TYPE' to define model type")

    model_type = cfg.TYPE

    if model_type == "cls":
        return build_model_from_cfg(cfg, CLASSIFIERS)
    elif model_type == "det":
        return build_model_from_cfg(cfg, DETECTORS)
    elif model_type == "emb":
        return build_model_from_cfg(cfg, EMBEDDERS)
    else:
        raise TypeError(f"No type for the task {model_type}")
