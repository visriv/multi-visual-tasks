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
class DetAnimalDataset(DetBaseDataset):
    """Data interface for the animal dataset"""

    CLASSES = (
        'golden-retriever', 'husky', 'teddy-dog', 'shiba-inu', 'samoyed',
        'corgi', 'alaskan', 'labrador', 'hyena', 'german-shepherd',
        'pomeranian', 'french-bulldog', 'bichon-frise', 'horse-dog',
        'tibetan-mastiff', 'border-collie', 'pug', 'chow-chow',
        'pit-bull', 'chihuahua', 'rottweiler', 'doberman', 'bull-terrier',
        'english-bulldog', 'akita', 'dachshund', 'caucasian-dog', 'bully',
        'giant-poodle', 'great-dane', 'schnauzer', 'teacup-dogs', 'shar-pei',
        'beagles', 'boxer', 'orange-cat', 'lynx', 'doll-cat',
        'british-shorthair-cat', 'folds', 'hairless-cat', 'caracal-cat',
        'tabby-cat', 'siamese-cat', 'persian-cat', 'american-shorthair',
        'maine-coon', 'russian-blue-cat', 'cow-cat', 'bengal-cat',
        'squirrel', 'fox', 'kangaroo', 'deer', 'zebra', 'wildebeest',
        'honey-badger', 'rhinoceros', 'giraffe', 'african-wild-dog', 'seal',
        'hedgehog', 'groundhog', 'donkey', 'brown-bear', 'black-bear',
        'polar-bear', 'grizzly-bear', 'raccoon', 'dinosaur', 'baboon',
        'warthog', 'camel', 'alpaca', 'koala', 'weasel', 'sloth', 'snow-leopard',
        'elk', 'rabbit', 'lion', 'giant-panda', 'tiger', 'monkey', 'elephant',
        'hamster', 'pig', 'cattle', 'sheep', 'horse', 'orangutan', 'antelope',
        'cheetah', 'wild-boar', 'fish', 'chickens', 'goose', 'penguin', 'hippo',
        'parrot', 'mouse', 'cathrow', 'longhair-cat', 'mini-doberman', 'garfield',
        'silver-gradient', 'short-footed-cat', 'otter', 'shetland', 'sika-deer')
    
    # 金毛犬,哈士奇,泰迪犬,柴犬,萨摩耶犬,
    # 柯基,阿拉斯加犬,拉布拉多犬,鬣狗,德国牧羊犬,
    # 博美犬,法国斗牛犬,比熊犬,马犬,
    # 藏獒,边境牧羊犬,巴哥犬,松狮犬,
    # 比特犬,吉娃娃,罗威纳犬,杜宾犬,牛头梗,
    # 英国斗牛犬,秋田犬,腊肠犬,高加索犬,恶霸犬,
    # 巨型贵宾犬,大丹犬,雪纳瑞,茶杯犬,沙皮犬,
    # 比格犬,拳师犬,橘猫,猞猁,布偶猫,
    # 英国短毛猫,折耳猫,无毛猫,狞猫,
    # 狸花猫,暹罗猫,波斯猫,美国短毛猫,
    # 缅因猫,俄罗斯蓝猫,奶牛猫,孟加拉猫,
    # 松鼠,狐狸,袋鼠,鹿,斑马,角马,
    # 蜜獾,犀牛,长颈鹿,非洲野狗,海豹,
    # 刺猬,土拨鼠,驴,棕熊,黑熊,
    # 北极熊,灰熊,浣熊,恐龙,狒狒,
    # 疣猪,骆驼,羊驼,考拉,黄鼠狼,树懒,雪豹,
    # 麋鹿,兔子,狮子,大熊猫,老虎,猴子,大象,
    # 仓鼠,猪,牛,羊,马,猩猩,羚羊,
    # 猎豹,野猪,鱼,鸡,鹅,企鹅,河马,
    # 鹦鹉,老鼠,卡斯罗犬,长毛猫,迷你杜宾,加菲猫,
    # 银渐层,矮脚猫,水獭,喜乐蒂,梅花鹿

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        """Same as base detection dataset"""

        super(DetAnimalDataset, self).__init__(
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
                    obj_info = obj_raw.split(' ')
                    if len(obj_info) < 5:
                        continue

                    if int(obj_info[0]) >= 0:
                        valid_inds.append(i)
                        break                
            else:
                valid_inds.append(i)

        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from yolo format.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        if not self.is_tfrecord:
            img_id = self.data_infos[idx]['id']
            width = self.data_infos[idx]['width']
            height = self.data_infos[idx]['height']
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
                obj_info = obj_raw.split(' ')
                if len(obj_info) < 5:
                    continue

                label = int(obj_info[0])
                assert label >= 0 and label < 110

                bbox_cx = float(obj_info[1]) * width
                bbox_cy = float(obj_info[2]) * height
                bbox_w = float(obj_info[3]) * width
                bbox_h = float(obj_info[4]) * height

                bbox = [
                    int(bbox_cx - bbox_w / 2 + 0.5),
                    int(bbox_cy - bbox_h / 2 + 0.5),
                    int(bbox_cx + bbox_w / 2 + 0.5),
                    int(bbox_cy + bbox_h / 2 + 0.5)
                ]

                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    if width < self.min_size or height < self.min_size:
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
            label = obj_cls[i]
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
