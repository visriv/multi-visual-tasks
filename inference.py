import json
import os
import pickle
import sys
from pathlib import Path
from collections import Counter
import numpy as np

if 'MVT_ROOT' in os.environ:
    MVT_ROOT = os.getenv('MVT_ROOT')
    print('Get MVT_ROOT: ', MVT_ROOT)
else:
    MVT_ROOT = str(Path(__file__).absolute().parent)
    os.environ['MVT_ROOT'] = MVT_ROOT
    print('Set MVT_ROOT: ', MVT_ROOT)
    sys.path.insert(0, MVT_ROOT)
    print('Add {} to PYTHONPATH'.format(MVT_ROOT))

import torch

from configs import cfg
from mvt.cores.metric_ops import LpDistance
from mvt.datasets.data_builder import build_dataloader, build_dataset
from mvt.models.model_builder import build_model
from mvt.utils.checkpoint_util import load_checkpoint
from mvt.utils.config_util import get_dataset_global_args, get_task_cfg
from mvt.utils.geometric_util import imresize
from mvt.utils.misc_util import ProgressBar
from mvt.utils.parallel_util import DataParallel
from mvt.utils.photometric_util import tensor2imgs


def det_single_device_test(model, data_loader, score_thr=0.05):
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

            single_res = model.module.show_result(
                img_show, result[i],
                show=False, bbox_color='red',
                text_color='red', out_file=None, score_thr=score_thr)

            img_names.append(img_meta['ori_filename'])
            results.append(single_res)

        for _ in range(batch_size):
            prog_bar.update()
    return {'img_names': img_names, 'detections': results}


def save_det_json(img_names, det_results, json_path):

    images = []
    annotations = []
    bbox_id = 0
    for i, det_result in enumerate(det_results):

        img_info = {
            "file_name": img_names[i],
            "id": i}
        images.append(img_info)
        for j in range(len(det_result)):
            anno_info = {
                "image_id": i,
                "id": bbox_id,
                "bbox": [
                    int(det_result[j, 0] + 0.5),
                    int(det_result[j, 1] + 0.5),
                    int(det_result[j, 2] - det_result[j, 0] + 0.5),
                    int(det_result[j, 3] - det_result[j, 1] + 0.5)],
                "category_id": det_result[j, 5],
                "score": det_result[j, 4]
            }
            annotations.append(anno_info)
            bbox_id += 1
    predictions = {"images": images, "annotations": annotations}

    with open(json_path, "w") as wf:
        json.dump(predictions, wf)

    print('Detections have been saved at {}'.format(json_path))


def emb_single_device_test(model, data_loader, with_label=False):
    model.eval()
    results = []
    labels = []
    bbox_ids = []

    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))

    for _, data in enumerate(data_loader):
        data['img_metas'] = data['img_metas'].data[0]
        data['img'] = data['img'].data[0]
        if with_label:
            data['label'] = data['label'].data[0]
        else:
            bbox_id_batch = data['bbox_id'].data.cpu().numpy()

        if 'bbox' in data:
            data['bbox'] = data['bbox'].data[0]

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        for i in range(batch_size):
            results.append(result[i])
            if with_label:
                labels.append(data['label'][i].data.cpu().numpy())
            else:
                bbox_ids.append(bbox_id_batch[i])

        for _ in range(batch_size):
            prog_bar.update()

    results = np.array(results)

    outputs = {'embeddings': results}

    if with_label:
        labels = np.array(labels)
        outputs['labels'] = labels
        cache_path = '/tmp/emb_ref.pkl'
    else:
        bbox_ids = np.array(bbox_ids)
        outputs['bbox_ids'] = bbox_ids
        cache_path = '/tmp/emb_qry.pkl'

    with open(cache_path, 'wb') as f:
        pickle.dump(outputs, f)

    return outputs


def run_det_task(cfg_path, model_path, json_path, score_thr):
    print('Running detection task ...')
    det_cfg = cfg.clone()
    get_task_cfg(det_cfg, cfg_path)

    # build the dataloader
    dataset_args = get_dataset_global_args(det_cfg.DATA)
    dataset = build_dataset(
        det_cfg.DATA.TEST_DATA, det_cfg.DATA.TEST_TRANSFORMS, dataset_args)
    data_loader = build_dataloader(
        dataset,
        samples_per_device=det_cfg.DATA.TEST_DATA.SAMPLES_PER_DEVICE,
        workers_per_device=det_cfg.DATA.TEST_DATA.WORKERS_PER_DEVICE,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(det_cfg.MODEL)

    checkpoint = load_checkpoint(model, model_path, map_location='cpu')

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = DataParallel(model, device_ids=[0])
    outputs = det_single_device_test(
        model, data_loader, score_thr=score_thr)

    save_det_json(outputs['img_names'], outputs['detections'], json_path)


def infer_labels(qry_outputs, ref_outputs, rank_list):
    """
    Get predicted labels

    Args:
        qry_outputs (dict): query outputs
        ref_outputs (dict): reference outputs
    Returns:
        outputs: query bbox indices with assigned labels
    """

    ref_emb = ref_outputs['embeddings']
    ref_labels = ref_outputs['labels']

    qry_emb = qry_outputs['embeddings']
    qry_ids = qry_outputs['bbox_ids']

    dist_func = LpDistance()
    if torch.cuda.is_available():
        ref_emb = torch.from_numpy(ref_emb).cuda()
        qry_emb = torch.from_numpy(qry_emb).cuda()

        mat = dist_func(qry_emb, ref_emb)
        mat = mat.data.cpu().numpy()
    else:
        ref_emb = torch.from_numpy(ref_emb)
        qry_emb = torch.from_numpy(qry_emb)

        mat = dist_func(qry_emb, ref_emb)
        mat = mat.data.numpy()

    mat_inds = np.argsort(mat, axis=1)

    ref_labels = ref_labels.reshape((ref_labels.shape[0], ))
    pred_labels = ref_labels[mat_inds]

    result = {}
    result['bbox_ids'] = qry_ids

    for k in rank_list:
        print('Assign label by frequency from top {} predictions'.format(k))
        if k == 1:
            result['labels_top_{}'.format(k)] = pred_labels[:, 0]
            continue

        pred_labels_top_k = []
        for i in range(pred_labels.shape[0]):
            top_k = pred_labels[i, :k]
            votes = Counter(top_k)
            pred = votes.most_common(1)[0][0]
            pred_labels_top_k.append(pred)

        result['labels_top_{}'.format(k)] = np.array(pred_labels_top_k)

    return result


def save_submit_json(outputs, det_json_path, out_json_path, score_thr=0.1, top_k=1):
    bbox_ids = outputs['bbox_ids']
    labels = outputs['labels_top_{}'.format(top_k)]
    pred_dict = {}
    for i, bbox_id in enumerate(bbox_ids):
        pred_dict[bbox_id] = labels[i]

    with open(det_json_path, 'r') as f:
        data_ori = json.load(f)

    ann_new = []
    for ann in data_ori['annotations']:
        assert ann['id'] in pred_dict
        if ann['score'] < score_thr:
            continue

        label = pred_dict[ann['id']]
        ann['category_id'] = int(label)
        ann_new.append(ann)

    data_ori['annotations'] = ann_new

    # out_json_path = out_json_path.replace('.json', '_top_{}.json'.format(top_k))
    with open(out_json_path, 'w') as wf:
        json.dump(data_ori, wf)
        print('Results have been saved at ' + out_json_path)


def run_emb_task(cfg_path, model_path, det_json_path,
                 out_json_path, score_thr, top_k_list=[1]):
    print('Running embedding task ...')
    emb_cfg = cfg.clone()
    get_task_cfg(emb_cfg, cfg_path)

    # build the dataloader
    dataset_args = get_dataset_global_args(emb_cfg.DATA)

    dataset_ref = build_dataset(
        emb_cfg.DATA.VAL_DATA, emb_cfg.DATA.TEST_TRANSFORMS, dataset_args)
    data_loader_ref = build_dataloader(
        dataset_ref,
        samples_per_device=emb_cfg.DATA.VAL_DATA.SAMPLES_PER_DEVICE,
        workers_per_device=emb_cfg.DATA.VAL_DATA.WORKERS_PER_DEVICE,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(emb_cfg.MODEL)

    checkpoint = load_checkpoint(model, model_path, map_location='cpu')

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset_ref.CLASSES

    model = DataParallel(model, device_ids=[0])

    outputs_ref = emb_single_device_test(
        model, data_loader_ref, with_label=True)

    dataset_qry = build_dataset(
        emb_cfg.DATA.TEST_DATA, emb_cfg.DATA.TEST_TRANSFORMS, dataset_args)
    data_loader_qry = build_dataloader(
        dataset_qry,
        samples_per_device=emb_cfg.DATA.TEST_DATA.SAMPLES_PER_DEVICE,
        workers_per_device=emb_cfg.DATA.TEST_DATA.WORKERS_PER_DEVICE,
        dist=False,
        shuffle=False)

    outputs_qry = emb_single_device_test(
        model, data_loader_qry, with_label=False)

    outputs = infer_labels(outputs_qry, outputs_ref, top_k_list)

    save_submit_json(
        outputs, det_json_path, out_json_path, score_thr, top_k=1)


def run():
    mvt_path = Path(MVT_ROOT)
    det_cfg_path = mvt_path / 'task_settings/img_det/det_yolov4_9a_retail_one.yaml'
    det_model_path = mvt_path / 'meta/train_infos/det_yolov4_9a_retail_one/epoch_200.pth'

    #det_cfg_path = mvt_path / 'task_settings/img_det/det_yolov4_retail_one.yaml'
    #det_model_path = mvt_path / 'meta/train_infos/det_yolov4_retail_one/epoch_200.pth'

    det_json_path = mvt_path / 'data/test/a_det_annotations.json'
    det_score_thr = 0.1

    run_det_task(str(det_cfg_path), str(det_model_path),
                 str(det_json_path), det_score_thr)

    emb_cfg_path = mvt_path / 'task_settings/img_emb/emb_resnet50_mlp_loc_retail.yaml'
    emb_model_path = mvt_path / 'meta/train_infos/emb_resnet50_mlp_loc_retail/epoch_50.pth'
    out_json_path = mvt_path / 'submit/out.json'

    run_emb_task(
        str(emb_cfg_path), str(emb_model_path),
        str(det_json_path), str(out_json_path),
        score_thr=det_score_thr, top_k_list=[1])


if __name__ == '__main__':
    run()
