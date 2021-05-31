# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 21:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com
# @File    : test.py
import argparse
import os
import torch
from yacs.config import CfgNode

from configs import cfg
from mvt.cores.ops import fuse_conv_bn
from mvt.utils.parallel_util import (
    DataParallel, DistributedDataParallel, init_dist)
from mvt.utils.misc_util import get_dist_info
from mvt.utils.checkpoint_util import load_checkpoint
from mvt.utils.fp16_util import wrap_fp16_model
from mvt.engines.evaluator import multi_device_test, single_device_test
from mvt.datasets.data_builder import build_dataloader, build_dataset
from mvt.models.model_builder import build_model
from mvt.utils.config_util import (
    get_task_cfg, get_dataset_global_args, get_dict_from_list, convert_to_dict)
from mvt.utils.io_util import obj_dump


def parse_args():
    parser = argparse.ArgumentParser(
        description='test (and eval) a model')
    parser.add_argument('task_config', help='test config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', 
        help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    get_task_cfg(cfg, args.task_config)

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

    # build the dataloader
    dataset_args = get_dataset_global_args(cfg.DATA)
    dataset = build_dataset(cfg.DATA.TEST_DATA, cfg.DATA.TEST_TRANSFORMS, dataset_args)
    data_loader = build_dataloader(
        dataset,
        samples_per_device=cfg.DATA.TEST_DATA.SAMPLES_PER_DEVICE,
        workers_per_device=cfg.DATA.TEST_DATA.WORKERS_PER_DEVICE,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(cfg.MODEL)
    if "FP16" in cfg.SCHEDULE:
        if isinstance(cfg.SCHEDULE.FP16, CfgNode): 
            # fp16_cfg = convert_to_dict(cfg.SCHEDULE.FP16)
            wrap_fp16_model(model)

    print('Loading checkpoint from: ', args.checkpoint)    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    # this walkaround is for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        if 'CLASSES' in dataset:
            model.CLASSES = dataset.CLASSES

    if not distributed:
        model = DataParallel(model, device_ids=[0])
        outputs = single_device_test(
            model, data_loader, model_type=cfg.MODEL.TYPE, 
            show=args.show, out_dir=args.show_dir,
            show_score_thr=args.show_score_thr)
    else:
        model = DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_device_test(
            model, data_loader, model_type=cfg.MODEL.TYPE, 
            tmpdir=args.tmpdir, gpu_collect=args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            obj_dump(outputs, args.out)
        kwargs = {}
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            if len(cfg.RUNTIME.EVALUATION) > 0:
                eval_kwargs = convert_to_dict(cfg.RUNTIME.EVALUATION)
            else:
                eval_kwargs = {}
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
