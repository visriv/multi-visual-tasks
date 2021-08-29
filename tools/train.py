# -*- coding: utf-8 -*-
# @Tme    : 2020/12/01 16:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com

import argparse
import os
import os.path as osp
import time
import sys
sys.path.append("..")

from configs import cfg
from mvt.utils.config_util import (get_task_cfg, convert_to_dict, 
                                   get_dataset_global_args,
                                   get_dict_from_list)
from mvt.utils.parallel_util import init_dist
from mvt.utils.log_util import get_root_logger
from mvt.utils.runtime_util import collect_env
from mvt.utils.runtime_util import set_random_seed
from mvt.models.model_builder import build_model
from mvt.datasets.data_builder import build_dataset
from mvt.engines.trainer import train_processor
from mvt.utils.path_util import PathManagerBase


def parse_args():
    """List the args for setting a task of training a model.

    Returns:
        args: Training settings.
    """
    parser = argparse.ArgumentParser(description="train a model")
    parser.add_argument("task_yaml_file", help="train setting file for configuration")
    parser.add_argument('--work-dir', default=None, help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', default=None, help='the checkpoint file to resume from')
    parser.add_argument(
        '--load-from', default=None, help='the checkpoint file to load from')
    parser.add_argument(
        '--no-test',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')

    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
        
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')

    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='rank id for multiprocessing')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)    

    return args


def main():
    """Main function for training."""
    args = parse_args()
    get_task_cfg(cfg, args.task_yaml_file)
    cfg.merge_from_list(args.opts)
    # print(cfg)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.RUNTIME.WORK_DIR = osp.join(
            args.work_dir,
            os.path.splitext(osp.basename(args.task_yaml_file))[0])
        print("Work path:", cfg.RUNTIME.WORK_DIR)
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.RUNTIME.WORK_DIR = osp.join(
            'meta/train_infos',
            os.path.splitext(osp.basename(args.task_yaml_file))[0])
    
    if args.resume_from is not None:
        cfg.RUNTIME.RESUME_MODEL_PATH = args.resume_from
    else:
        cfg.RUNTIME.RESUME_MODEL_PATH = ""
    
    if args.load_from is not None:
        cfg.RUNTIME.LOAD_CHECKPOINT_PATH = args.load_from
    else:
        cfg.RUNTIME.LOAD_CHECKPOINT_PATH = ""

    if args.gpu_ids is not None:
        cfg.RUNTIME.GPU_IDS = args.gpu_ids
    else:
        cfg.RUNTIME.GPU_IDS = "" if args.gpus is None else list(range(args.gpus))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if "DIST_PARAMS" in cfg.RUNTIME:
            if isinstance(cfg.RUNTIME.DIST_PARAMS, list):
                init_dist(
                    args.launcher,
                    **get_dict_from_list(cfg.RUNTIME.DIST_PARAMS))
        else:
            init_dist(args.launcher)

    # create work_dir
    if (cfg.RUNTIME.WORK_DIR is not None and 
            cfg.RUNTIME.WORK_DIR != ""):        
        cfg.RUNTIME.WORK_DIR = osp.expanduser(
            cfg.RUNTIME.WORK_DIR)
        os.makedirs(cfg.RUNTIME.WORK_DIR, mode=0o777, exist_ok=True)
    if args.seed is not None:
        cfg.RUNTIME.SEED = args.seed
    cfg.freeze()
    # dump config using functions from fvcore
    dump_yaml_path = osp.join(
        cfg.RUNTIME.WORK_DIR,
        osp.basename(args.task_yaml_file))
    PathManager = PathManagerBase()
    with PathManager.open(dump_yaml_path, "w") as f:
        f.write(cfg.dump())
    
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.RUNTIME.WORK_DIR, f'{timestamp}.log')

    logger = get_root_logger(
        log_file=log_file, log_level=cfg.RUNTIME.LOG_LEVEL)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'

    meta['env_info'] = env_info
    meta['config'] = convert_to_dict(cfg)

    if 'RANK' in os.environ:
        if os.environ['RANK'] == 0:
            logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                        dash_line)
            # log some basic info
            logger.info(f'Distributed training: {distributed}')
            logger.info(f'Config:\n{cfg}')
    else:
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg}')

    # set random seeds
    if args.seed is not None:
        if 'RANK' in os.environ:
            if os.environ['RANK'] == 0:
                logger.info(f'Set random seed to {args.seed}, '
                            f'deterministic: {args.deterministic}')
        else:
            logger.info(f'Set random seed to {args.seed}, '
                        f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.task_yaml_file)

    model = build_model(cfg.MODEL)

    dataset_args = get_dataset_global_args(cfg.DATA)

    train_dataset = build_dataset(cfg.DATA.TRAIN_DATA, cfg.DATA.TRAIN_TRANSFORMS, dataset_args)    
    if len(cfg.RUNTIME.WORKFLOW) == 2:
        val_dataset = build_dataset(cfg.DATA.VAL_DATA, cfg.DATA.TRAIN_TRANSFORMS, dataset_args)
    else:
        val_dataset = None
    
    if not args.no_test:
        test_dataset = build_dataset(cfg.DATA.TEST_DATA, cfg.DATA.TEST_TRANSFORMS, dataset_args)
    else:
        test_dataset = None

    # print(model.state_dict().keys())

    train_processor(
        cfg,
        model,
        train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        distributed=distributed,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()

