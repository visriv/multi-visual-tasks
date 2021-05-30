# -*- coding: utf-8 -*-
# @Time    : 2020/11/16 19:00
# @Author  : zhiming.qian
# @Email   : zhimingqian@tencent.com
# @File    : mosaic.py

import numpy as np
from PIL import Image
from io import BytesIO
import os.path as osp

from .det_base import DetBaseDataset
from mtl.datasets.data_wrapper import DATASETS


@DATASETS.register_module()
class MosaicDataset(DetBaseDataset):
    """Data interface for the Mosaic dataset"""

    CLASSES = ('mosaic')

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        """Same as base detection dataset"""

        super(MosaicDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, sel_index)
            
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}

        if "MIN_SIZE" in data_cfg:
            self.min_size = data_cfg.MIN_SIZE
        else:
            self.min_size = None

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        data_infos = []
        with open(ann_file, 'r') as f:
            lines = f.readlines()
        anno_list = [line.strip() for line in lines if line.strip()!='']

        for anno_info in anno_list:
            img_path = osp.join(self.data_prefix, anno_info + '.jpg')
            # label_path = osp.join(self.anno_prefix, anno_info + '.txt')
            img = Image.open(img_path)
            width, height = img.size
            
            data_infos.append(
                dict(id=anno_info, filename=img_path, width=width, height=height))

        return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""

        if self.is_tfrecord:
            raise ValueError("It should not be used when taking tfrecord.")

        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue

            if self.filter_empty_gt:
                img_id = img_info['id']
                label_path = osp.join(self.anno_prefix, img_id + '.txt')
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    obj_raw = line.strip()
                    if obj_raw == '':
                        continue
                    obj_info = obj_raw.split('\t')
                    if len(obj_info) < 5:
                        continue

                    if int(obj_info[0]) == 1:
                        valid_inds.append(i)
                        break                
            else:
                valid_inds.append(i)

        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        if not self.is_tfrecord:
            img_id = self.data_infos[idx]['id']
            label_path = osp.join(self.anno_prefix, img_id + '.txt')
            
            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []

            with open(label_path, 'r') as f:
                lines = f.readlines()        
            for line in lines:
                obj_raw = line.strip()
                if obj_raw == '':
                    continue
                obj_info = obj_raw.split('\t')
                if len(obj_info) < 5:
                    continue
                if int(obj_info[0]) != 1:
                    continue
                label = 0

                bbox = [
                    int(float(obj_info[1]) + 0.5),
                    int(float(obj_info[2]) + 0.5),
                    int(float(obj_info[3]) + 0.5),
                    int(float(obj_info[4]) + 0.5)
                ]

                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True

                if ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)
            
            if not bboxes:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0, ))
            else:
                bboxes = np.array(bboxes, ndmin=2)
                labels = np.array(labels)
            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 4))
                labels_ignore = np.zeros((0, ))
            else:
                bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
                labels_ignore = np.array(labels_ignore)
            ann = dict(
                bboxes=bboxes.astype(np.float32),
                labels=labels.astype(np.int64),
                bboxes_ignore=bboxes_ignore.astype(np.float32),
                labels_ignore=labels_ignore.astype(np.int64))
            return ann
        else:
            return self.getitem_info(idx)['ann']

    def record_parser(self, feature_list):
        """Call when is_tfrecord is ture.
        
        feature_list = [(key, feature), (key, feature)]
        key is your label.txt col name
        feature is oneof bytes_list, int64_list, float_list
        """

        for key, feature in feature_list:

            #for image file col
            if key == 'name':
                img_id = feature.bytes_list.value[0].decode('UTF-8','strict')
            if key == 'image':
                image_raw = feature.bytes_list.value[0]
                pil_img = Image.open(BytesIO(image_raw)).convert('RGB')
                img = np.array(pil_img).astype(np.float32)
            elif key == 'bbox/class':   
                obj_cls = feature.int64_list.value
            elif key == 'bbox/xmin':    
                obj_xmin = feature.int64_list.value
            elif key == 'bbox/ymin':  
                obj_ymin = feature.int64_list.value
            elif key == 'bbox/xmax': 
                obj_xmax = feature.int64_list.value
            elif key == 'bbox/ymax':   
                obj_ymax = feature.int64_list.value
        
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []

        for i in range(len(obj_cls)):
            if obj_cls[i] != 1:
                continue
            label = 0
            bbox = [
                int(float(obj_xmin[i]) + 0.5),
                int(float(obj_ymin[i]) + 0.5),
                int(float(obj_xmax[i]) + 0.5),
                int(float(obj_ymax[i]) + 0.5)
            ]

            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True

            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        
        return {
            'img_info': dict(id=img_id, 
                             filename=img_id + '.jpg', 
                             width=pil_img.size[0], 
                             height=pil_img.size[1]),
            'img': img, 
            'ann': ann
        }
