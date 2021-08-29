import copy
import inspect

import torch

from mvt.utils.reg_util import Registry, build_module_from_dict
from mvt.utils.config_util import convert_to_dict

OPTIMIZERS = Registry("optimizer")
OPTIMIZER_BUILDERS = Registry("optimizer builder")


def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith("__"):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()


def build_optimizer_constructor(optimizer_dict):
    return build_module_from_dict(optimizer_dict, OPTIMIZER_BUILDERS)


def build_optimizer(model, cfg):
    optimizer_dict = convert_to_dict(cfg)
    constructor_type = optimizer_dict.pop("CONSTRUCTOR", "DefaultOptimizerConstructor")
    paramwise_dict = optimizer_dict.pop("PARAMWISE_CFG", None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_dict,
            paramwise_cfg=paramwise_dict,
        )
    )
    optimizer = optim_constructor(model)
    return optimizer
