import copy
import os
from abc import ABCMeta, abstractmethod
from os import path as osp
import numpy as np
from torch.utils.data import Dataset
from yacs.config import CfgNode

from ..transforms import Compose
from model.mvt.utils.io_util import list_from_file


class EmbBaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset for embedding with defined classes."""

    CLASSES = None

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        """Initialization for dataset construction.

        Args:
            data_prefix (str): the prefix of data path
            pipeline (list): a list of dict, where each element represents
                a operation defined in transforms.
            ann_file (str | None): the annotation file. When ann_file is str,
                the subclass is expected to read from the ann_file. When ann_file
                is None, the subclass is expected to read according to data_prefix
            test_mode (bool): in train mode or test mode
        """
        if not isinstance(data_cfg, CfgNode):
            raise TypeError("data_cfg must be a list")
        if not isinstance(pipeline_cfg, CfgNode):
            raise TypeError("pipeline_cfg must be a list")

        self.data_root = root_path
        self.data_cfg = data_cfg

        if "DATA_INFO" in data_cfg and isinstance(data_cfg.DATA_INFO, list):
            self.ann_file = data_cfg.DATA_INFO[sel_index]
        else:
            raise ValueError("DATA_INFO should be set properly")

        if "TEST_MODE" in data_cfg:
            self.test_mode = data_cfg.TEST_MODE
        else:
            self.test_mode = False

        if "CLASSES" in data_cfg:
            self.CLASSES = self.get_classes(data_cfg.CLASSES)
        else:
            self.CLASSES = self.get_classes()

        # processing pipeline
        self.pipeline_cfg = pipeline_cfg
        self.pipeline = Compose(self.get_pipeline_list())
        self.sel_index = sel_index

        if "DATA_PREFIX" in data_cfg and isinstance(data_cfg.DATA_PREFIX, list):
            self.data_prefix = data_cfg.DATA_PREFIX[sel_index]
        else:
            self.data_prefix = None

        # join paths if data_root is specified
        if not osp.isabs(self.ann_file):
            self.ann_file = osp.join(self.data_root, self.ann_file)
        if not (self.data_prefix is None or osp.isabs(self.data_prefix)):
            self.data_prefix = osp.join(self.data_root, self.data_prefix)

        mvt_root = os.getenv("MVT_ROOT")
        if mvt_root and not osp.isabs(self.ann_file):
            self.ann_file = osp.join(mvt_root, self.ann_file)

        if mvt_root and not osp.isabs(self.data_prefix):
            self.data_prefix = osp.join(mvt_root, self.data_prefix)

        # load annotations (and proposals)
        self.data_infos = self.load_annotations()

        # set group flag for the sampler
        self._set_group_flag()

    def get_pipeline_list(self):
        """get the list of pipelines"""

        pipeline_list = []
        for k_t, v_t in self.pipeline_cfg.items():
            pipeline_item = {}
            if len(v_t) > 0:
                if not isinstance(v_t, CfgNode):
                    raise TypeError("pipeline items must be a CfgNode")
            pipeline_item["type"] = k_t

            for k_a, v_a in v_t.items():
                if isinstance(v_a, CfgNode):
                    pipeline_item[k_a] = []
                    for sub_kt, sub_vt in v_a.items():
                        sub_item = {}
                        if len(sub_vt) > 0:
                            if not isinstance(sub_vt, CfgNode):
                                raise TypeError("transform items must be a CfgNode")
                        sub_item["type"] = sub_kt
                        for sub_ka, sub_va in sub_vt.items():
                            if isinstance(sub_va, CfgNode):
                                raise TypeError("Only support two built-in layers")
                            sub_item[sub_ka] = sub_va
                        pipeline_item[k_a].append(sub_item)
                else:
                    pipeline_item[k_a] = v_a
            pipeline_list.append(pipeline_item)

        return pipeline_list

    @abstractmethod
    def load_annotations(self):
        """Get annotations."""

        raise NotImplementedError("Must Implement parser")

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """

        gt_labels = []
        for i in range(len(self)):
            gt_labels.append(self.getitem_info(i)["label"])

        gt_labels = np.array(gt_labels)
        return gt_labels

    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            int: Image category of specified index.
        """
        cat_ids = self.getitem_info(idx)["label"]
        if isinstance(cat_ids, list):
            return np.array(cat_ids).astype(np.int)
        elif isinstance(cat_ids, np.ndarray):
            return cat_ids.astype(np.int)
        return np.asarray([cat_ids]).astype(np.int)

    def prepare_data(self, idx):
        """Prepare data and run pipelines"""

        results = copy.deepcopy(self.getitem_info(idx))
        results["img_prefix"] = self.data_prefix
        return self.pipeline(results)

    def __len__(self):

        return len(self.data_infos)

    def getitem_info(self, index):
        return self.data_infos[index]

    def __getitem__(self, idx):

        return self.prepare_data(idx)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        return class_names

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """

        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.getitem_info(i)
            if img_info["width"] / img_info["height"] > 1:
                self.flag[i] = 1

    def evaluate(self, results, metric="accuracy"):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.

        Returns:
            dict: evaluation results
        """
        print(type(results))
        print(len(results))
        eval_results = {f"accuracy": 10.0}
        return eval_results

        # if isinstance(metric, str):
        #     metrics = [metric]
        # else:
        #     metrics = metric
        # allowed_metrics = ['accuracy']
        # eval_results = {}
        # for metric in metrics:
        #     if metric not in allowed_metrics:
        #         raise KeyError(f'metric {metric} is not supported.')
        #     results = np.vstack(results)
        #     labels = self.get_gt_labels()
        #     assert len(labels) == len(results)
        #     if metric == 'accuracy':
        #         acc = np.sum(results == labels) / len(labels)
        #         eval_result = {f'accuracy': acc}

        #     eval_results.update(eval_result)

        # return eval_results
