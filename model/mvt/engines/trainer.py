import random
import numpy as np
import torch
from yacs.config import CfgNode

from model.mvt.utils.parallel_util import CustomDataParallel
from model.mvt.utils.parallel_util import CustomDistributedDataParallel
from model.mvt.cores.core_hook import HOOKS
from model.mvt.cores.hook import (
    DistSamplerSeedHook,
    OptimizerHook,
    EvalHook,
    DistEvalHook,
)
from model.mvt.cores.runner import EpochBasedRunner
from model.mvt.cores.core_optimizer import build_optimizer
from model.mvt.utils.log_util import get_root_logger
from model.mvt.datasets.data_builder import build_dataloader
from model.mvt.utils.data_util import replace_ImageToTensor
from model.mvt.utils.reg_util import build_module_from_dict
from model.mvt.utils.config_util import convert_to_dict


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_builtin_config_dict(log_cfg):
    log_config = {}
    for k_t, v_t in log_cfg.items():
        if isinstance(v_t, CfgNode):
            log_config[k_t] = []
            for sub_kt, sub_vt in v_t.items():
                sub_item = {}
                if len(sub_vt) > 0:
                    if not isinstance(sub_vt, CfgNode):
                        raise TypeError("transform items must be a CfgNode")
                sub_item["type"] = sub_kt
                for sub_ka, sub_va in sub_vt.items():
                    if isinstance(sub_va, CfgNode):
                        raise TypeError("Only support two built-in layers")
                    sub_item[sub_ka] = sub_va

                log_config[k_t].append(sub_item)
        else:
            log_config[k_t] = v_t
    return log_config


def train_processor(
    cfg,
    model,
    dataset,
    val_dataset=None,
    test_dataset=None,
    distributed=False,
    timestamp=None,
    meta=None,
):
    """Excute training process with cfg, model, dataset and so on.
    Args:
        cfg (CfgNode): Config for task.
        model (torch.nn.Module): Model for training.
        dataset (torch.utils.data.Dataset): Training dataset.
        val_dataset (torch.utils.data.Dataset): Validating dataset. Optional.
        distributed (bool): Whether using distributed training.
        timestamp (str): String type of time information.
        meta (dict): Additional training information.
    """
    logger = get_root_logger(cfg.RUNTIME.LOG_LEVEL)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("FIND_UNUSED_PARAMETERS", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.cuda())
        model = CustomDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        if cfg.RUNTIME.GPU_IDS != "":
            model = CustomDataParallel(
                model.cuda(cfg.RUNTIME.GPU_IDS[0]), device_ids=cfg.RUNTIME.GPU_IDS
            )
        else:
            model = CustomDataParallel(model, device_ids=None)

    # build runner
    optimizer = build_optimizer(model, cfg.SCHEDULE.OPTIMIZER)
    fp16 = cfg.SCHEDULE.OPTIMIZER_CONFIG.get("fp16", False)
    print("Set fp16 as: ", fp16)
    runner = EpochBasedRunner(
        model,
        fp16=fp16,
        optimizer=optimizer,
        work_dir=cfg.RUNTIME.WORK_DIR,
        logger=logger,
        meta=meta,
    )
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # optimizer setting
    optim_config_dict = convert_to_dict(cfg.SCHEDULE.OPTIMIZER_CONFIG)
    for item in optim_config_dict:
        if optim_config_dict[item] == "":
            optim_config_dict[item] = None
    if "type" not in optim_config_dict:
        optimizer_config = OptimizerHook(**optim_config_dict)
    else:
        optimizer_config = optim_config_dict

    # register hooks
    lr_config_dict = convert_to_dict(cfg.SCHEDULE.LR_POLICY)
    checkpoint_config = convert_to_dict(cfg.RUNTIME.CHECKPOINT_CONFIG)
    log_config = get_builtin_config_dict(cfg.RUNTIME.LOG_CONFIG)
    if "MOMENTUM_CONFIG" in cfg.RUNTIME:
        momentum_config = cfg.RUNTIME.MOMENTUM_CONFIG
    else:
        momentum_config = None
    runner.register_training_hooks(
        lr_config_dict, optimizer_config, checkpoint_config, log_config, momentum_config
    )
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # prepare data loaders
    train_dataloader = build_dataloader(
        dataset,
        cfg.DATA.TRAIN_DATA.SAMPLES_PER_DEVICE,
        cfg.DATA.TRAIN_DATA.WORKERS_PER_DEVICE,
        # cfg.gpus will be ignored if distributed
        cfg.RUNTIME.GPU_IDS,
        dist=distributed,
        seed=cfg.RUNTIME.SEED,
    )
    if val_dataset is not None:
        val_dataloader = build_dataloader(
            val_dataset,
            cfg.DATA.VAL_DATA.SAMPLES_PER_DEVICE,
            cfg.DATA.VAL_DATA.WORKERS_PER_DEVICE,
            # cfg.gpus will be ignored if distributed
            cfg.RUNTIME.GPU_IDS,
            dist=distributed,
            seed=cfg.RUNTIME.SEED,
        )
        data_loaders = [train_dataloader, val_dataloader]
    else:
        data_loaders = [train_dataloader]

    # register eval hooks
    if test_dataset is not None:
        test_dataloader = build_dataloader(
            test_dataset,
            cfg.DATA.TEST_DATA.SAMPLES_PER_DEVICE,
            cfg.DATA.TEST_DATA.WORKERS_PER_DEVICE,
            dist=distributed,
            shuffle=False,
        )
        # Support batch_size > 1 in validation
        val_samples_per_device = cfg.DATA.TEST_DATA.SAMPLES_PER_DEVICE
        if val_samples_per_device > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.defrost()
            replace_ImageToTensor(cfg.DATA.TEST_TRANSFORMS)
            cfg.freeze()
        if len(cfg.RUNTIME.EVALUATION) > 0:
            eval_cfg = convert_to_dict(cfg.RUNTIME.EVALUATION)
        else:
            eval_cfg = {}
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(test_dataloader, cfg.MODEL.TYPE, **eval_cfg))

    # user-defined hooks
    if len(cfg.RUNTIME.CUSTOM_HOOKS) > 0:
        for hook_cfg in cfg.RUNTIME.CUSTOM_HOOKS:
            custom_hook = convert_to_dict(hook_cfg)
            assert isinstance(custom_hook, dict), (
                "Each item in custom_hooks expects dict type, but got "
                f"{type(custom_hook)}"
            )
            item_cfg = custom_hook.copy()
            priority = item_cfg.pop("priority", "NORMAL")
            hook = build_module_from_dict(item_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.RUNTIME.RESUME_MODEL_PATH != "":
        runner.resume(cfg.RUNTIME.RESUME_MODEL_PATH)
    elif cfg.RUNTIME.LOAD_CHECKPOINT_PATH != "":
        runner.load_checkpoint(cfg.RUNTIME.LOAD_CHECKPOINT_PATH)

    runner.run(data_loaders, cfg.RUNTIME.WORKFLOW, cfg.SCHEDULE.TOTAL_EPOCHS)
