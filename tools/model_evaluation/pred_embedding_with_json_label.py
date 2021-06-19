# -*- coding: utf-8 -*-
import argparse
import json
import os.path as osp
import pickle

import numpy as np
import torch

from configs import cfg
from mvt.cores.metric_ops import CosineSimilarity, LpDistance
from mvt.datasets.data_builder import build_dataloader, build_dataset
from mvt.models.model_builder import build_model
from mvt.utils.checkpoint_util import load_checkpoint
from mvt.utils.config_util import get_dataset_global_args, get_task_cfg
from mvt.utils.misc_util import ProgressBar
from mvt.utils.parallel_util import DataParallel

RANK_LIST = [1, 3, 5, 10, 20, 50, 100]


def single_device_test(model, data_loader):
    model.eval()
    results = []
    bbox_ids = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))

    for _, data in enumerate(data_loader):
        data['img_metas'] = data['img_metas'].data[0]
        data['img'] = data['img'].data[0]
        bbox_id_batch = data['bbox_id'].data.cpu().numpy()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        for i in range(batch_size):
            results.append(result[i])
            bbox_ids.append(bbox_id_batch[i])

        for _ in range(batch_size):
            prog_bar.update()

    results = np.array(results)
    bbox_ids = np.array(bbox_ids)
    outputs = {'embeddings': results, 'bbox_ids': bbox_ids}
    with open('/tmp/out.pkl', 'wb') as f:
        pickle.dump(outputs, f)
    return outputs


def infer_labels(outputs, ref_file):
    """
    Get predicted labels

    Args:
        outputs: dict output from embedding model
        ref_file: pickle file containing reference embeddings and labels
    Returns:
        outputs: bbox indices with assigned labels
    """
    with open(ref_file, 'rb') as f:
        ref_data = pickle.load(f)

    ref_emb = ref_data['embeddings']
    ref_labels = ref_data['labels']

    qry_emb = outputs['embeddings']
    qry_ids = outputs['bbox_ids']

    dist_func = LpDistance()
    ref_emb = torch.from_numpy(ref_emb).cuda()
    qry_emb = torch.from_numpy(qry_emb).cuda()

    mat = dist_func(qry_emb, ref_emb)
    mat = mat.data.cpu().numpy()
    mat_inds = np.argsort(mat, axis=1)

    ref_labels = ref_labels.reshape((ref_labels.shape[0],))
    pred_labels = ref_labels[mat_inds]

    result = {}
    result['bbox_ids'] = qry_ids

    for k in RANK_LIST:
        print('Assign label by frequency from top {} predictions'.format(k))
        if k == 1:
            result['labels_top_{}'.format(k)] = pred_labels[:, 0]
            continue

        pred_labels_top_k = []
        for i in range(pred_labels.shape[0]):
            pred = np.argmax(np.bincount(pred_labels[i, :k]))
            pred_labels_top_k.append(pred)

        result['labels_top_{}'.format(k)] = np.array(pred_labels_top_k)

    return result


def save_json(outputs, json_ori, json_out, top_k=1):
    bbox_ids = outputs['bbox_ids']
    labels = outputs['labels_top_{}'.format(top_k)]
    pred_dict = {}
    for i, bbox_id in enumerate(bbox_ids):
        pred_dict[bbox_id] = labels[i]

    with open(json_ori, 'r') as f:
        data_ori = json.load(f)

    for ann in data_ori['annotations']:
        assert ann['id'] in pred_dict
        label = pred_dict[ann['id']]
        ann['category_id'] = int(label)

    json_out = json_out.replace('.json', '_top_{}.json'.format(top_k))
    with open(json_out, 'w') as wf:
        json.dump(data_ori, wf)
        print('Result has been saved at ' + json_out)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run embedding model on testing data and output final json file')
    parser.add_argument('task_config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('reference', help='reference embedding file')
    parser.add_argument(
        '--json-ori', required=True, help='path of json file output from detection')
    parser.add_argument(
        '--json-out', required=True, help='path to save final submition file')
    args = parser.parse_args()

    return args


def main():
    """
    Example:
    python3 tools/model_evaluation/pred_embedding_with_json_label.py \
            task_settings/img_emb/emb_resnet50_fc_retail.yaml \
            meta/emb_resnet50_fc_retail/epoch_50.pth meta/reference_test_b_embedding.pkl \
            --json-ori data/test/a_det_annotations.json \
            --json-out submit/out.json
    """
    args = parse_args()

    get_task_cfg(cfg, args.task_config)

    # build the dataloader
    dataset_args = get_dataset_global_args(cfg.DATA)
    dataset = build_dataset(
        cfg.DATA.TEST_DATA, cfg.DATA.TEST_TRANSFORMS, dataset_args)
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
    outputs = single_device_test(model, data_loader)

    #with open('/tmp/out.pkl', 'rb') as f:
    #    outputs = pickle.load(f)

    outputs = infer_labels(outputs, args.reference)

    for k in RANK_LIST:
        save_json(outputs, args.json_ori, args.json_out, top_k=k)


if __name__ == '__main__':
    main()
