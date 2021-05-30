import functools
import warnings
import torch
import inspect
import torch.nn as nn
from torch.nn import SyncBatchNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

TORCH_VERSION = torch.__version__


def deprecated_api_warning(name_dict, cls_name=None):
    """A decorator to check if some argments are deprecate and try to replace
    deprecate src_arg_name to dst_arg_name.
    Args:
        name_dict(dict):
            key (str): Deprecate argument names.
            val (str): Expected argument names.
    Returns:
        func: New function.
    """

    def api_warning_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # get the arg spec of the decorated method
            args_info = inspect.getfullargspec(old_func)
            # get name of the function
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f'{cls_name}.{func_name}'
            if args:
                arg_names = args_info.args[:len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead')
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead')
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            # apply converted arguments to the decorated method
            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper

def infer_abbr(class_type):
    """Infer abbreviation from the class name.
    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.
    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".
    Args:
        class_type (type): The norm layer type.
    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm'

def is_norm(layer, exclude=None):
    """Check if a layer is a normalization layer.
    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type | tuple[type]): Types to be excluded.
    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not isinstance(exclude, tuple):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)
