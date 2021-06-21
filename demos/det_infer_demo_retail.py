# -*- coding: utf-8 -*-
# @Time    : 2020/3/16 18:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com

import os
import sys
import json

from configs import cfg
from mvt.utils.config_util import get_task_cfg
from mvt.engines.predictor import (get_detector, inference_detector,
                                   show_detector_result)

# 漏检 745 374 1022 1026 193 1271 982 1308 452 574 84
# 重检 745 1151 314 489 1450 352 903 841 305 311 1038 1224 321 1044 574 84
# 误检 1021 521 958
# 定位不准 253 450


def load_image_infos(json_file):
    with open(str(json_file), 'r') as f:
        infos = json.load(f)

    img_infos = infos['images']
    return img_infos


if __name__ == '__main__':
    """Test demo for detection models"""

    index = int(sys.argv[1])

    # task_config_path = 'task_settings/img_det/det_yolov4_retail_one.yaml'
    # checkpoint_path = 'meta/train_infos/det_yolov4_retail_one/epoch_200.pth'
    task_config_path = 'task_settings/img_det/det_yolov4x_cspdarknet_retail_one.yaml'
    checkpoint_path = 'meta/train_infos/det_yolov4x_cspdarknet_retail_one/epoch_60.pth'

    with_show = True
    show_score_thr = 0.01

    print(
        'Infer the detection results from image with index {}.'.format(index))
    assert index >= 0

    image_dir = 'data/RetailDet/test/a_images/'
    json_file = 'data/RetailDet/test/a_annotations.json'

    img_infos = load_image_infos(json_file)
    image_file_name = img_infos[index]['file_name']
    image_path = os.path.join(image_dir, image_file_name)

    save_dir = 'meta/test_res'
    save_path = os.path.join(save_dir, image_file_name)  # None for no saving

    # get config
    get_task_cfg(cfg, task_config_path)

    # get model
    model = get_detector(cfg, checkpoint_path, device='cpu')

    # get result
    result = inference_detector(cfg, model, image_path)

    # show and save result
    show_detector_result(model, image_path, result, show_score_thr, with_show,
                         save_path)
