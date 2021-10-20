import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from yacs.config import CfgNode

from mvt.cores.eval.common_eval import eval_map, eval_recalls
from mvt.datasets.data_wrapper import DATASETS
from mvt.datasets.transforms import Compose
from mvt.utils.io_util import file_load, list_from_file


@DATASETS.register_module()
class D3dBaseDataset(Dataset):
    """Base dataset for detection."""

    class_names = None

    def __init__(self, data_cfg, pipeline_cfg, root_path, net=None, sel_index=0):
        """Initialization for dataset construction

        Args:
            data_cfg (cfgNode): dataset info.
            pipeline_cfg (cfgNode): Processing pipeline info.
            root_path (str, optional): Data root for ``ann_file``, ``data_prefix``,
                ``seg_prefix``, ``proposal_file`` if specified.
            sel_index (int): select the annotation file with the index from
                annotation list.
        """

        if not isinstance(data_cfg, CfgNode):
            raise TypeError("data_cfg must be a list")
        if not isinstance(pipeline_cfg, CfgNode):
            raise TypeError("pipeline_cfg must be a list")
        if "DATA_INFO" not in data_cfg:
            raise AttributeError("data_cfg should have node DATA_INFO")
        if not isinstance(data_cfg.DATA_INFO, list):
            raise TypeError("data_cfg.DATA_INFO must be a list")

        self.data_root = root_path
        self.data_cfg = data_cfg
        self.ann_file = data_cfg.DATA_INFO[sel_index]
        self.net = net

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        if "FILTER_EMPTY_GT" in data_cfg:
            self.filter_empty_gt = data_cfg.FILTER_EMPTY_GT
        else:
            self.filter_empty_gt = True

        if "TEST_MODE" in data_cfg:
            self.test_mode = data_cfg.TEST_MODE
        else:
            self.test_mode = False

        if "CLASS_NAMES" in data_cfg:
            self.class_names = self.get_classes(data_cfg.CLASS_NAMES)
        else:
            self.class_names = self.get_classes()
        self.cat2id = {name: i for i, name in enumerate(self.class_names)}

        self.sel_index = sel_index
        # processing pipeline
        self.pipeline_cfg = pipeline_cfg
        self.pipeline = Compose(self.get_pipeline_list())

        # load annotations (and proposals)
        if "DATA_PREFIX" in data_cfg and isinstance(data_cfg.DATA_PREFIX, list):
            self.data_prefix = data_cfg.DATA_PREFIX[sel_index]
        else:
            self.data_prefix = None

        # join paths if data_root is specified
        if not (self.data_prefix is None or osp.isabs(self.data_prefix)):
            self.data_prefix = osp.join(self.data_root, self.data_prefix)

        # only use ann_file[0]
        self.ann_file = self.ann_file[0]
        if not osp.isabs(self.ann_file):
            self.ann_file = osp.join(self.data_root, self.ann_file)
        self.data_infos = self.load_annotations(self.ann_file)

    def __len__(self):
        """Total number of samples of data."""

        return len(self.data_infos)

    def get_pipeline_list(self):
        """Get the list of configures for constructing pipelines

        Note:
            self.pipeline is a CfgNode

        Returns:
            list[dict]: list of dicts with types and parameters for
                constructing pipelines.
        """

        pipeline_list = []
        for k_t, v_t in self.pipeline_cfg.items():
            pipeline_item = {}
            if len(v_t) > 0:
                if not isinstance(v_t, CfgNode):
                    raise TypeError("pipeline items must be a CfgNode")

            pipeline_item["type"] = k_t

            for k_a, v_a in v_t.items():
                if isinstance(v_a, CfgNode):
                    if "type" in v_a:
                        pipeline_item[k_a] = {}
                        for sub_kt, sub_vt in v_a.items():
                            pipeline_item[k_a][sub_kt] = sub_vt
                    else:
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

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""

        return file_load(ann_file)

    def getitem_info(self, index):

        return self.data_infos[index]

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = np.random.choice(len(self.data_infos))
                continue
            return data

    def get_sensor_data(self, idx):
        raise NotImplementedError("This function should be overrided")

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        data_info = self.getitem_info(idx)
        results = self.get_sensor_data(data_info)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

        data_info = self.getitem_info(idx)
        results = self.get_sensor_data(data_info)
        results["net"] = self.net
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default class_names defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the class_names defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """

        if classes is None:
            return cls.class_names

        if isinstance(classes, str):
            # take it as a file path
            class_names = list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        return class_names

    def evaluate(
        self,
        results,
        metric="mAP",
        logger=None,
        iou_thr=0.5,
        scale_ranges=None,
    ):
        pass
