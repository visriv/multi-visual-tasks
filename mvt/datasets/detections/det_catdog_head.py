import os.path as osp
import numpy as np

from .det_animal import DetAnimalDataset
from mtl.datasets.data_wrapper import DATASETS


@DATASETS.register_module()
class CatDogHeadDataset(DetAnimalDataset):
    """Data interface for the Mosaic dataset"""

    CLASSES = ('cat', 'dog')

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

                label = int(obj_info[0])

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
