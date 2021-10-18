import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from yacs.config import CfgNode

from mvt.cores.eval.common_eval import eval_map, eval_recalls
from mvt.datasets.data_wrapper import DATASETS
from mvt.datasets.transforms import Compose
from mvt.utils.io_util import file_load, list_from_file


@DATASETS.register_module()
class DetBaseDataset(Dataset):
    """Base dataset for detection."""

    class_names = None

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
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

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.getitem_info(idx)["ann"]

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""

        results["img_prefix"] = self.data_prefix
        results["seg_prefix"] = self.seg_prefix
        results["proposal_file"] = self.proposal_file
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["dataset"] = self

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

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.getitem_info(idx)
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info, _idx=idx)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

        img_info = self.getitem_info(idx)
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results["proposals"] = self.proposals[idx]
        self.pre_pipeline(results)
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

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

        pass

    def evaluate(
        self,
        results,
        metric="mAP",
        logger=None,
        proposal_nums=(100, 300, 1000),
        iou_thr=0.5,
        scale_ranges=None,
    ):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ["mAP", "recall"]
        if metric not in allowed_metrics:
            raise KeyError(f"metric {metric} is not supported")
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == "mAP":
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.class_names,
                logger=logger,
            )
            eval_results["mAP"] = mean_ap
        elif metric == "recall":
            gt_bboxes = [ann["bboxes"] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger
            )
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f"recall@{num}@{iou}"] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f"AR@{num}"] = ar[i]

        return eval_results
