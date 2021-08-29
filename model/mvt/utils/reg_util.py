# -*- coding: utf-8 -*-
# @Time    : 2020/12/02 16:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com
# @File    : reg_util.py

import inspect

from .config_util import convert_to_dict


class Registry:
    """A registry to map strings to classes.
    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = (
            self.__class__.__name__ + f"(name={self._name}, "
            f"items={self._module_dict})"
        )
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the registry record.
        Args:
            key (str): The class name in string format.
        Returns:
            class: The corresponding class.
        """
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, " f"but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered " f"in {self.name}")

        self._module_dict[module_name] = module_class

    def register_module(self, name=None, force=False, module=None):
        """Register a module.
        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.
        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)
        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(f"name must be a str, but got {type(name)}")

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register


def build_data_from_cfg(data_cfg, pipeline_cfg, default_args, registry, sel_index=0):
    """Build a dataset from config dict.
    Args:
        data_cfg (cfgNode): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
    Returns:
        object: The constructed object.
    """
    if not isinstance(registry, Registry):
        raise TypeError(
            "registry must be an Registry object, " f"but got {type(registry)}"
        )
    if "type" not in default_args:
        raise KeyError("cfg must have a name to define dataset class")

    data_cls_name = default_args["type"]

    if isinstance(data_cls_name, str):
        data_cls = registry.get(data_cls_name)
        if data_cls is None:
            raise KeyError(f"{data_cls_name} is not in the {registry.name} registry")
    else:
        raise TypeError(f"type must be a str type, but got {type(data_cls_name)}")

    if "root_path" not in default_args:
        raise KeyError("args must have a root path to find the dataset")

    return data_cls(data_cfg, pipeline_cfg, default_args["root_path"], sel_index)


def build_model_from_cfg(cfg, registry):
    """Build a model from config dict.
    Args:
        cfg (cfgNode): Config node. It should at least contain the node 'NAME'.
        registry (:obj:`Registry`): The registry to search the 'NAME' from.
    Returns:
        object: The constructed object.
    """
    if not isinstance(registry, Registry):
        raise TypeError(
            "registry must be an Registry object, " f"but got {type(registry)}"
        )
    if "NAME" not in cfg:
        raise KeyError("cfg must have a name to define class")

    model_cls_name = cfg.NAME

    if isinstance(model_cls_name, str):
        model_cls = registry.get(model_cls_name)
        if model_cls is None:
            raise KeyError(f"{model_cls_name} is not in the {registry.name} registry")
    else:
        raise TypeError(f"NAME must be a str type, but got {type(model_cls_name)}")

    return model_cls(cfg)


def build_module_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should contain the key 'type'.
        registry (:obj:`Registry`): The registry to search the type from.
    Returns:
        object: The constructed object.
    """
    if not isinstance(registry, Registry):
        raise TypeError(
            "registry must be an Registry object, " f"but got {type(registry)}"
        )
    if "type" not in cfg:
        raise KeyError("cfg must have a type to define class")

    obj_type = cfg.type

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

    cfg_dict = convert_to_dict(cfg)
    cfg_dict.pop("type")

    if default_args is not None and len(default_args) > 0:
        return obj_cls(**cfg_dict, **default_args)
    else:
        return obj_cls(**cfg_dict)


def build_module_from_dict(cfg_dict, registry, default_args=None):
    """Constructor with the setting of config dict.
    Args:
        cfg_dict (dict): Config dict. It should at least contain the node 'type'.
        registry (:obj:`Registry`): The registry to search the type from.
    Returns:
        object: The constructed object.
    """
    if not isinstance(registry, Registry):
        raise TypeError(
            "registry must be an Registry object, " f"but got {type(registry)}"
        )
    if "type" not in cfg_dict:
        raise KeyError("cfg_dict must have a name to define module")

    module_cls_name = cfg_dict["type"]

    if isinstance(module_cls_name, str):
        module_cls = registry.get(module_cls_name)
        if module_cls is None:
            raise KeyError(f"{module_cls_name} is not in the {registry.name} registry")
    else:
        raise TypeError(f"type must be a str type, but got {type(module_cls_name)}")

    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            "default_args must be a dict or None, " f"but got {type(default_args)}"
        )

    cfg_dict.pop("type")
    if default_args is not None and len(default_args) > 0:
        return module_cls(**cfg_dict, **default_args)
    else:
        return module_cls(**cfg_dict)
