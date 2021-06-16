# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 21:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com

import argparse
import os.path as osp
import torch
import json

from configs import cfg
from mvt.utils.parallel_util import DataParallel
from mvt.utils.checkpoint_util import load_checkpoint
from mvt.datasets.data_builder import build_dataloader, build_dataset
from mvt.models.model_builder import build_model
from mvt.utils.config_util import get_task_cfg, get_dataset_global_args
from mvt.utils.geometric_util import imresize
from mvt.utils.misc_util import ProgressBar
from mvt.utils.photometric_util import tensor2imgs


def single_device_det_test(model,
                           data_loader,
                           show=False,
                           out_dir=None,
                           show_score_thr=0.3):
    model.eval()
    results = []
    img_names = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))

    for _, data in enumerate(data_loader):
        for i in range(len(data['img_metas'])):
            data['img_metas'][i] = data['img_metas'][i].data[0]
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        
        if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
            img_tensor = data['img'][0]
        else:
            img_tensor = data['img'][0].data
        img_metas = data['img_metas'][0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = imresize(img_show, (ori_w, ori_h))

            if out_dir:
                out_file = osp.join(out_dir, img_meta['ori_filename'])
            else:
                out_file = None

            single_res = model.module.show_result(
                img_show,
                result[i],
                show=show,
                bbox_color='red',
                text_color='red',
                out_file=out_file,
                score_thr=show_score_thr)
            img_names.append(img_meta['ori_filename'])
            results.append(single_res)

        for _ in range(batch_size):
            prog_bar.update()
    return {'img_names': img_names, 'detections': results}


def single_device_test_vis(model,
                           data_loader,
                           model_type='det',
                           show=False,
                           out_dir=None,
                           show_score_thr=0.3):
    if model_type == 'det':
        return single_device_det_test(model, data_loader, show, out_dir, show_score_thr)
    else:
        return None


def save_json(img_names, det_results, json_path):

    images = []
    annotations = []
    for i, det_result in enumerate(det_results):

        img_info = {
            "file_name": img_names[i], 
            "id": i}
        images.append(img_info)
        for j in range(len(det_result)):
            anno_info = {
                "image_id": i,
                "bbox": [
                    int(det_result[j, 0] + 0.5), 
                    int(det_result[j, 1] + 0.5), 
                    int(det_result[j, 2] - det_result[j, 0] + 0.5), 
                    int(det_result[j, 3] - det_result[j, 1] + 0.5)],
                "category_id": det_result[j, 5],
                "score": det_result[j, 4]
            }
            annotations.append(anno_info)
    predictions = {"images":images, "annotations":annotations}

    with open(json_path, "w") as wf:
        json.dump(predictions, wf)


def parse_args():
    parser = argparse.ArgumentParser(description='test a model')
    parser.add_argument('task_config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--out-dir', help='directory where painted images will be saved')
    parser.add_argument('--json-path', help='json path to save labels')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    assert args.show or args.out_dir, \
        ('Please specify at least one operation (save/eval/show the '
         'results / save the results) with the argument "--show" or "--out-dir"')

    get_task_cfg(cfg, args.task_config)

    # build the dataloader
    dataset_args = get_dataset_global_args(cfg.DATA)
    dataset = build_dataset(cfg.DATA.TEST_DATA, cfg.DATA.TEST_TRANSFORMS, dataset_args)
    data_loader = build_dataloader(
        dataset,
        samples_per_device=cfg.DATA.TEST_DATA.SAMPLES_PER_DEVICE,
        workers_per_device=cfg.DATA.TEST_DATA.WORKERS_PER_DEVICE,
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
    outputs = single_device_test_vis(
        model, data_loader, show=args.show, out_dir=args.out_dir,
        show_score_thr=args.show_score_thr)
    
    if args.json_path:
        save_json(outputs['img_names'], outputs['detections'], args.json_path)


if __name__ == '__main__':
    main()
