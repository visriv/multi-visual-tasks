# -*- coding: utf-8 -*-
# @Time    : 2020/3/16 18:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com

import os

from configs import cfg
from mvt.utils.config_util import get_task_cfg
from mvt.engines.predictor import (get_detector, inference_detector,
                                   show_detector_result)
    

if __name__ == '__main__':
    """Test demo for detection models"""

    print('Infer the detection results from an image.')
    image_file_name = '9.jpg'
    image_dir = 'meta/test_data'
    image_path = os.path.join(image_dir, image_file_name)
    save_dir = 'meta/test_res'
    save_path = os.path.join(save_dir, image_file_name) # None for no saving

    task_config_path = 'task_settings/img_det/det_efficient_d7_retail.yaml'
    checkpoint_path = 'meta/train_infos/det_efficient_d7_retail/epoch_18.pth'
        
    with_show = True
    show_score_thr = 0.05
    
    # get config
    get_task_cfg(cfg, task_config_path)
    
    # get model
    model = get_detector(cfg, checkpoint_path, device='cpu')
 
    # get result
    result = inference_detector(cfg, model, image_path)
    
    # show and save result
    show_detector_result(
        model, image_path, result, show_score_thr, with_show, save_path)
