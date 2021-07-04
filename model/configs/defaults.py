# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 19:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com
# @File    : default.py

from yacs.config import CfgNode as CN


# --------------------------------------------------------------------------- #
# Root handle for all configs
# --------------------------------------------------------------------------- #
_C = CN()

# --------------------------------------------------------------------------- #
# Model configs, including backbone, neck, multiple heads and extend for others
# --------------------------------------------------------------------------- #
_C.MODEL = CN()

# Submodels
_C.MODEL.BACKBONE = CN() # backbone settings
_C.MODEL.NECK = CN() # neck settings
_C.MODEL.CLS_HEAD = CN()
_C.MODEL.EMB_HEAD = CN()
_C.MODEL.BBOX_HEAD = CN()
_C.MODEL.DECODE_HEAD = CN()
_C.MODEL.AUXILIARY_HEAD = CN()
_C.MODEL.TRAIN_CFG = CN()
_C.MODEL.TEST_CFG = CN()
_C.MODEL.RPN_HEAD = CN()
_C.MODEL.ROI_HEAD = CN()
_C.MODEL.LOSS = CN()
_C.MODEL.EXTEND = CN()

# --------------------------------------------------------------------------- #
# Data settings, including data abstractions, contents and transforms
# --------------------------------------------------------------------------- #
_C.DATA = CN()

# Data structures and pipelines
_C.DATA.TRAIN_TRANSFORMS = CN()
_C.DATA.TEST_TRANSFORMS = CN()
_C.DATA.TRAIN_DATA = CN()
_C.DATA.VAL_DATA = CN()
_C.DATA.TEST_DATA= CN()

# ---------------------------------------------------------------------------- #
# Schedule methods, including optimizers and learning policies
# ---------------------------------------------------------------------------- #
_C.SCHEDULE = CN()
# train configs
_C.SCHEDULE.OPTIMIZER = CN()
_C.SCHEDULE.OPTIMIZER_CONFIG = CN()
_C.SCHEDULE.LR_POLICY = CN()

# ---------------------------------------------------------------------------- #
# Runtime settings, including hooks, logs, evaluations and so on
# ---------------------------------------------------------------------------- #
_C.RUNTIME = CN()
_C.RUNTIME.CHECKPOINT_CONFIG = CN()
_C.RUNTIME.LOG_CONFIG = CN()
_C.RUNTIME.CUSTOM_HOOKS = CN()
_C.RUNTIME.EVALUATION = CN()
