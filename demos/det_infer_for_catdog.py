# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 18:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com
import os

from configs import cfg
from mtl.utils.config_util import get_task_cfg
from mtl.utils.vis_util import show_det_result
from mtl.engines.predictor import get_detector, inference_detector
from mtl.utils.io_util import write_det_xml


if __name__ == '__main__':
    """Test demo for detection models"""

    img_dir = '/Users/qianzhiming/Desktop/data/other-datasets/CoverCrop/images'
    label_dir = '/Users/qianzhiming/Desktop/data/other-datasets/CoverCrop/catdog_labels'
    
    task_config_path = 'task_settings/img_det/det_yolov3_resnet_catdoghead.yaml'
    checkpoint_path = 'meta/train_infos/det_yolov3_resnet_catdoghead/epoch_100.pth'
        
    with_show = True
    show_score_thr = 0.3
    
    # get config
    get_task_cfg(cfg, task_config_path)    
    # get model
    model = get_detector(cfg, checkpoint_path, device='cpu')

    for img_sub_dir in os.listdir(img_dir):
        if not img_sub_dir.startswith('.') and os.path.isdir(
                os.path.join(img_dir, img_sub_dir)):
            print('Processing with: ', img_sub_dir)
            for img_file in os.listdir(
                    os.path.join(img_dir, img_sub_dir)):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    img_path = os.path.join(img_dir, img_sub_dir, img_file)
                    label_path = os.path.join(
                            label_dir, img_sub_dir, 
                            img_file[:-3] + 'xml')
                    if os.path.isfile(label_path):
                        continue
                    
                        # get result
                    result = inference_detector(cfg, model, img_path)

                    # show and save result
                    bboxes, labels, height, width = show_det_result(
                        img_path, result, 'CatDogHeadDataset', show_score_thr, show=with_show)
                    
                    write_det_xml(label_path, width, height, bboxes, labels, 'CatDogHeadDataset')
