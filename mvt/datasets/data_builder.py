# -*- coding: utf-8 -*-
# @Time    : 2020/12/03 16:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com

import os
import glob
import resource
from functools import partial
from torch.utils.data import DataLoader, RandomSampler

from .data_sampler import DistributedGroupSampler, DistributedSampler, GroupSampler
from .data_wrapper import ConcatDataset, RepeatDataset, ClassBalancedDataset, DATASETS
from mvt.utils.reg_util import build_data_from_cfg
from mvt.utils.runtime_util import worker_init_fn, collate
from mvt.utils.misc_util import get_dist_info


# --------------------------------------------------------------------------- #
# Avoid resource limit
# ---------------------------------------------------------------------------
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
hard_limit = rlimit[1]
soft_limit = min(4096, hard_limit)
resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def build_dataset(data_cfg, pipeline_cfg, default_args, sel_index=0):
    if isinstance(data_cfg, (list, tuple)):
        dataset = ConcatDataset(
            [build_dataset(c, pipeline_cfg, default_args, sel_index) for c in data_cfg]
        )
    elif data_cfg.TYPE == "Concat":
        cfg_opt = data_cfg.clone()
        cfg_opt.defrost()
        cfg_opt.TYPE = "Normal"
        dataset = ConcatDataset(
            [
                build_dataset(cfg_opt, pipeline_cfg, default_args, sel_index=i)
                for i in range(len(data_cfg.DATA_INFO))
            ],
            data_cfg.FLAG,
        )
    elif data_cfg.TYPE == "Repeat":
        cfg_opt = data_cfg.clone()
        cfg_opt.defrost()
        cfg_opt.TYPE = "Concat"
        dataset = RepeatDataset(
            build_dataset(cfg_opt, pipeline_cfg, default_args, sel_index), data_cfg.FLAG
        )
    elif data_cfg.TYPE == "Balanced":
        cfg_opt = data_cfg.clone()
        cfg_opt.defrost()
        cfg_opt.TYPE = "Concat"
        dataset = ClassBalancedDataset(
            build_dataset(cfg_opt, pipeline_cfg, default_args, sel_index), data_cfg.FLAG
        )
    else:
        relative_path_list = []
        for relative_path in data_cfg.DATA_INFO[sel_index]:
            if "*" in relative_path:
                extend_list = glob.glob(
                    os.path.join(default_args["root_path"], relative_path)
                )
                len_root_path = len(default_args["root_path"])
                for extend_global_path in extend_list:
                    re_path = extend_global_path[len_root_path:]
                    if re_path.startswith("/"):
                        re_path = re_path[1:]
                    relative_path_list.append(re_path)
            else:
                relative_path_list.append(relative_path)
        data_cfg.DATA_INFO[sel_index] = relative_path_list
        dataset = build_data_from_cfg(
            data_cfg, pipeline_cfg, default_args, DATASETS, sel_index
        )

    return dataset


def build_dataloader(
    dataset,
    samples_per_device,
    workers_per_device,
    gpu_ids=None,
    dist=False,
    shuffle=True,
    seed=None,
    **kwargs
):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_device (int): Number of training samples on each device,
            i.e., batch size of each device.
        workers_per_device (int): How many subprocesses to use for data
            loading for each device.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle and hasattr(dataset, "flag"):
            sampler = DistributedGroupSampler(
                dataset, samples_per_device, world_size, rank
            )
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=True
            )  # if dataset size is large enough, shuffle can be set as False
        batch_size = samples_per_device
        num_workers = workers_per_device
    else:
        if hasattr(dataset, "flag"):
            sampler = GroupSampler(dataset, samples_per_device) if shuffle else None
        else:
            sampler = RandomSampler(dataset) if shuffle else None

        if gpu_ids is not None and gpu_ids != "":
            num_gpus = len(gpu_ids)
            batch_size = num_gpus * samples_per_device
            num_workers = num_gpus * workers_per_device
        else:
            batch_size = samples_per_device
            num_workers = workers_per_device

    init_fn = (
        partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs
    )

    return data_loader
