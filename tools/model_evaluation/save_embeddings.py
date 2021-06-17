# -*- coding: utf-8 -*-
import argparse
import os.path as osp
import pickle

import numpy as np
import torch

from configs import cfg
from mvt.datasets.data_builder import build_dataloader, build_dataset
from mvt.models.model_builder import build_model
from mvt.utils.checkpoint_util import load_checkpoint
from mvt.utils.config_util import get_dataset_global_args, get_task_cfg
from mvt.utils.misc_util import ProgressBar
from mvt.utils.parallel_util import DataParallel


def single_device_emb_test(model, data_loader, save_path=None):
    model.eval()
    results = []
    labels = []
    img_names = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))

    for _, data in enumerate(data_loader):
        data['img_metas'] = data['img_metas'].data[0]
        data['img'] = data['img'].data[0]
        data['label'] = data['label'].data[0]

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        for i in range(batch_size):
            results.append(result[i])
            labels.append(data['label'][i].data.cpu().numpy())

        for _ in range(batch_size):
            prog_bar.update()

    results = np.array(results)
    labels = np.array(labels)
    print(results.shape)
    print(labels.shape)
    outputs = {'embeddings': results, 'labels': labels}

    with open(save_path, "wb") as wf:
        pickle.dump(outputs, wf)
        print('Ebeddings have been saved at {}'.format(save_path))


def single_device_test(model, data_loader, save_path=None):
    return single_device_emb_test(model, data_loader, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='test a model')
    parser.add_argument('task_config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--save-path', required=True, help='path to save embeddings (and labels)')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    get_task_cfg(cfg, args.task_config)

    # build the dataloader
    dataset_args = get_dataset_global_args(cfg.DATA)
    dataset = build_dataset(
        cfg.DATA.VAL_DATA, cfg.DATA.TEST_TRANSFORMS, dataset_args)
    data_loader = build_dataloader(
        dataset,
        samples_per_device=cfg.DATA.VAL_DATA.SAMPLES_PER_DEVICE,
        workers_per_device=cfg.DATA.VAL_DATA.WORKERS_PER_DEVICE,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(cfg.MODEL)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = DataParallel(model, device_ids=[0])
    outputs = single_device_test(
        model, data_loader, save_path=args.save_path)


if __name__ == '__main__':
    main()
