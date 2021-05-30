import os.path as osp
import warnings
import struct
import numpy as np
from torch.utils.data import Dataset
from yacs.config import CfgNode
from threading import local

from mtl.utils.gen_tfrecords.yt_example_pb2 import Example
from mtl.cores.eval.common_eval import eval_map, eval_recalls
from mtl.datasets.data_wrapper import DATASETS
from mtl.datasets.transforms import Compose
from mtl.utils.io_util import file_load, list_from_file
from mtl.utils.tfrecord_util import tfrecord2idx


@DATASETS.register_module()
class DetBaseDataset(Dataset):
    """Base dataset for detection."""

    CLASSES = None

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
        
        if "TEST_MODE" in data_cfg:
            self.test_mode = data_cfg.TEST_MODE
        else:
            self.test_mode = False

        if "FILTER_EMPTY_GT" in data_cfg:
            self.filter_empty_gt = data_cfg.FILTER_EMPTY_GT
        else:
            self.filter_empty_gt = True
        
        if "CLASSES" in data_cfg:
            self.CLASSES = self.get_classes(data_cfg.CLASSES)
        else:
            self.CLASSES = self.get_classes()
        
        self.sel_index = sel_index
        self.is_tfrecord = data_cfg.IS_TFRECORD
        # processing pipeline
        self.pipeline_cfg = pipeline_cfg
        self.pipeline = Compose(self.get_pipeline_list())
        
        if self.is_tfrecord: # using self.idxs instead of self.data_infos
            # for tfrecord, the data should be full after filtering
            self.data_prefix = None
            self.proposals = None
            self.anno_prefix = None
            self.seg_prefix = None
            self.proposal_file = None

            tfindexs = []
            for i in range(len(self.ann_file)):
                if not osp.isabs(self.ann_file[i]):
                    self.ann_file[i] = osp.join(
                        self.data_root, self.ann_file[i])

                tfindexs.append(tfrecord2idx(
                    self.ann_file[i], 
                    self.ann_file[i].replace('.tfrecord', '.idx')))

            self.idxs = []
            self.tffiles = None
            self.samples = 0
            for index, tffile in zip(tfindexs, self.ann_file):
                idx = []
                with open(index) as idxf:
                    for line in idxf:
                        offset, _ = line.split(' ')
                        idx.append(offset)
                self.samples += len(idx)
                print("load %s, samples:%s" %(tffile,  len(idx)))
                self.idxs.append((idx, tffile))
        else:
            # load annotations (and proposals)
            if ("DATA_PREFIX" in data_cfg and 
                    isinstance(data_cfg.DATA_PREFIX, list)):
                self.data_prefix = data_cfg.DATA_PREFIX[sel_index]
            else:
                self.data_prefix = None

            if ("ANNO_PREFIX" in data_cfg and 
                    isinstance(data_cfg.ANNO_PREFIX, list)):
                self.anno_prefix = data_cfg.ANNO_PREFIX[sel_index]
            else:
                self.anno_prefix = None

            if ("SEG_PREFIX" in data_cfg and 
                    isinstance(data_cfg.SEG_PREFIX, list)):
                self.seg_prefix = data_cfg.SEG_PREFIX[sel_index]
            else:
                self.seg_prefix = None
            
            if ("PROPOSAL_FILE" in data_cfg and 
                    isinstance(data_cfg.PROPOSAL_FILE, list)):
                self.proposal_file = data_cfg.PROPOSAL_FILE[sel_index]
            else:
                self.proposal_file = None

            # join paths if data_root is specified
            if not (self.data_prefix is None or osp.isabs(self.data_prefix)):
                self.data_prefix = osp.join(self.data_root, self.data_prefix)
            if not (self.anno_prefix is None or osp.isabs(self.anno_prefix)):
                self.anno_prefix = osp.join(self.data_root, self.anno_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root, self.proposal_file)
            if self.proposal_file is not None:
                self.proposals = self.load_proposals(self.proposal_file)
            else:
                self.proposals = None
            
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            self.data_infos = self.load_annotations(self.ann_file)
            
            # filter images too small and containing no annotations
            if not self.test_mode:
                valid_inds = self._filter_imgs()
                self.data_infos = [self.data_infos[i] for i in valid_inds]
                if self.proposals is not None:
                    self.proposals = [self.proposals[i] for i in valid_inds]
        
        # set group flag for the sampler
        self._set_group_flag()

    def __len__(self):
        """Total number of samples of data."""

        if self.is_tfrecord:
            return self.samples
        else:
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
            if self.is_tfrecord and k_t == 'LoadImageFromFile':
                # remove the load image process
                continue
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

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""

        return file_load(ann_file)

    def load_proposals(self, proposal_file):
        """Load proposal from proposal file."""

        return file_load(proposal_file)

    def record_parser(self, feature_list):
        """Call when is_tfrecord is ture."""

        raise NotImplementedError("Must Implement parser")
    
    def get_record(self, f, offset):
        """Get the record when is_tfrecord is true."""
        
        if not self.is_tfrecord:
            raise ValueError(
                "Please set is_tfrecord to be true when call this function")

        f.seek(offset)

        # length,crc
        byte_len_crc = f.read(12)
        proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
        # proto,crc
        pb_data = f.read(proto_len)
        if len(pb_data) < proto_len:
            print("read pb_data err,proto_len:%s pb_data len:%s"%(proto_len, len(pb_data)))
            return None
        
        example = Example()
        example.ParseFromString(pb_data)
        #keep key value in order
        feature = sorted(example.features.feature.items())
     
        record = self.record_parser(feature)
        #print(record)
        return record
    
    def getitem_info(self, index):
        if self.is_tfrecord:
            if self.tffiles is None:
                self.tffiles = dict()
                for idx, tffile in self.idxs:
                    self.tffiles[tffile] = open(tffile, 'rb')

            for idx, tffile in self.idxs:
                if index >= len(idx):
                    index -= len(idx)
                    continue
                # every thread keep a f instace
                f = self.tffiles[tffile]

                offset = int(idx[index])
                return  self.get_record(f, offset)

            print("bad index,", index)
        else:
            return self.data_infos[index]

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.getitem_info(idx)['ann']
        
    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        
        return self.get_ann_info(idx)['labels'].astype(np.int).tolist()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""

        results['img_prefix'] = self.data_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""

        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        
        self.flag = np.zeros(len(self), dtype=np.uint8)
        if not self.is_tfrecord:
            self.flag = np.zeros(len(self), dtype=np.uint8)
            for i in range(len(self)):
                img_info = self.getitem_info(i)
                if img_info['width'] / img_info['height'] > 1:
                    self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""

        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

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
                idx = self._rand_another(idx)
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
        if self.is_tfrecord:
            item_info = self.getitem_info(idx)
            results = {
                'img_info': item_info['img_info'],
                'ann_info': item_info['ann'],
                'filename': item_info['img_info']['filename'],
                'ori_filename': item_info['img_info']['id'],
                'img': item_info['img'],
                'img_shape': item_info['img'].shape,
                'ori_shape': item_info['img'].shape,
                'img_fields': ['img']
            } # no need to load image
        else:
            img_info = self.getitem_info(idx)
            ann_info = self.get_ann_info(idx)
            results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
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
        if self.is_tfrecord:
            item_info = self.getitem_info(idx)
            results = {
                'img_info': item_info['img_info'],
                'filename': item_info['img_info']['filename'],
                'ori_filename': item_info['img_info']['id'],
                'img': item_info['img'],
                'img_shape': item_info['img'].shape,
                'ori_shape': item_info['img'].shape,
                'img_fields': ['img']
            } # no need to load image
        else:
            img_info = self.getitem_info(idx)
            results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

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
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        
        pass

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
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
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    
        return eval_results
