from functools import partial

import torch

TORCH_VERSION = torch.__version__
from model.mvt.utils.reg_util import Registry
from torch.nn.parallel import DataParallel, DistributedDataParallel

MODULE_WRAPPERS = Registry('module wrapper')
MODULE_WRAPPERS.register_module(module=DataParallel)
MODULE_WRAPPERS.register_module(module=DistributedDataParallel)


def is_module_wrapper(module):
    """Check if a module is a module wrapper.
    module wrappers: DataParallel, DistributedDataParallel.
     
    Args:
        module (nn.Module): The module to be checked.
    Returns:
        bool: True if the input module is a module wrapper.
    """
    module_wrappers = tuple(MODULE_WRAPPERS.module_dict.values())
    return isinstance(module, module_wrappers)
