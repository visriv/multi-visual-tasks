# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 21:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com

import logging
import os
from pathlib import Path
import yaml
from yacs.config import CfgNode


def merge_dict_into_cfg(cfg, meta_dict):
    """Merge dict into cfg"""
    if not isinstance(meta_dict, dict):
        raise TypeError(
            f"meta_dict should be a dict, but got {type(meta_dict)}")
    for dict_key in meta_dict:
        if isinstance(meta_dict[dict_key], dict):
            if dict_key not in cfg:
                cfg[dict_key] = CfgNode()
                # raise AttributeError(
                #     f"cfg has no attribute {dict_key}")
            merge_dict_into_cfg(cfg[dict_key], meta_dict[dict_key])
        else:
            cfg[dict_key] = meta_dict[dict_key]


def merge_file_into_cfg(cfg, meta_yaml_file):
    """Merge yaml file into cfg"""
    with open(meta_yaml_file) as fb:
        meta_data = yaml.load(fb)
    merge_dict_into_cfg(cfg, meta_data)


def _dict_update(meta_dict, additional_dict, task_data, mode):
    """Update two dicts for task file"""
    additional_dict[mode] = {}
    for data_key in task_data[mode]:
        if data_key == "BASE":
            mvt_root = Path(os.getenv('MVT_ROOT', './'))
            cfg_path = mvt_root / 'model/configs' / task_data[mode]["BASE"]
            with open(str(cfg_path)) as fd:
                data_info = yaml.load(fd, Loader=yaml.FullLoader)
            meta_dict[mode] = data_info[mode]
        else:
            additional_dict[mode][data_key] = task_data[mode][data_key]


def update_meta_dict(dst_dict, src_dict):
    """Update the meta dict by removing meaningless items"""
    for item_key in src_dict:
        if item_key not in dst_dict:
            print(item_key, dst_dict)
            raise AttributeError("The item is not in dst_dict!")
        if isinstance(src_dict[item_key], dict):
            if not isinstance(dst_dict[item_key], dict):
                raise TypeError(f"The item in dst_dict should be a dict, "
                                f"but got {type(dst_dict[item_key])}")
            update_meta_dict(dst_dict[item_key], src_dict[item_key])
        else:
            dst_dict[item_key] = src_dict[item_key]


def get_task_cfg(cfg, task_yaml_file):
    """Get the cfg from the task yaml file"""
    with open(task_yaml_file) as fb:
        task_data = yaml.load(fb, Loader=yaml.FullLoader)

    meta_dict = {}
    additional_dict = {}
    for task_key in task_data:
        if not isinstance(task_data[task_key], dict):
            raise TypeError("The first level attribute in task file "
                            "should be a dict!")
        if task_key == "MODEL":
            _dict_update(meta_dict, additional_dict, task_data, "MODEL")
        elif task_key == "DATA":
            _dict_update(meta_dict, additional_dict, task_data, "DATA")
        elif task_key == "SCHEDULE":
            _dict_update(meta_dict, additional_dict, task_data, "SCHEDULE")
        elif task_key == "RUNTIME":
            _dict_update(meta_dict, additional_dict, task_data, "RUNTIME")
        else:
            raise AttributeError("Unknown attribute in task file!")
    # print("raw dict:", meta_dict)
    # print("add dict:", additional_dict)
    update_meta_dict(meta_dict, additional_dict)
    # print("merge dict:", meta_dict)
    merge_dict_into_cfg(cfg, meta_dict)


def convert_to_dict(cfg_node, key_list=[]):
    """Convert cfg to dict"""
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
    return cfg_dict


def get_dataset_global_args(cfg):
    """Get the global settings for the training and testing dataset"""
    global_info = {}
    global_info["type"] = cfg.NAME
    global_info["root_path"] = cfg.ROOT_PATH
    return global_info


def _assert_with_logging(cond, msg):
    if not cond:
        logging.debug(msg)
    assert cond, msg


def get_dict_from_list(src_list):
    """Get dict from a list. 
    For example, `src_list = ['a', 0.5]`.
    """
    _assert_with_logging(
        len(src_list) % 2 == 0,
        "Override list has odd length: {}; it must be a list of pairs".format(
            src_list
        ),
    )
    dst_dict = {}
    for src_key, src_value in zip(src_list[0::2], src_list[1::2]):
        dst_dict[src_key] = src_value

    return dst_dict
