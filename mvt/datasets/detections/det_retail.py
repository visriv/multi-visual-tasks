import numpy as np

from mvt.utils.io_util import file_load
from mvt.datasets.data_wrapper import DATASETS
from mvt.datasets.detections.det_base import DetBaseDataset
from mvt.cores.eval.common_eval import eval_map, eval_recalls


@DATASETS.register_module()
class DetRetailDataset(DetBaseDataset):

    CLASSES = (
        "asamu",
        "baishikele",
        "baokuangli",
        "aoliao",
        "bingqilinniunai",
        "chapai",
        "fenda",
        "guolicheng",
        "haoliyou",
        "heweidao",
        "hongniu",
        "hongniu2",
        "hongshaoniurou",
        "kafei",
        "kaomo_gali",
        "kaomo_jiaoyan",
        "kaomo_shaokao",
        "kaomo_xiangcon",
        "kele",
        "laotansuancai",
        "liaomian",
        "lingdukele",
        "maidong",
        "mangguoxiaolao",
        "moliqingcha",
        "niunai",
        "qinningshui",
        "quchenshixiangcao",
        "rousongbing",
        "suanlafen",
        "tangdaren",
        "wangzainiunai",
        "weic",
        "weitanai",
        "weitaningmeng",
        "wulongcha",
        "xuebi",
        "xuebi2",
        "yingyangkuaixian",
        "yuanqishui",
        "xuebi-b",
        "kebike",
        "tangdaren3",
        "chacui",
        "heweidao2",
        "youyanggudong",
        "baishikele-2",
        "heweidao3",
        "yibao",
        "kele-b",
        "AD",
        "jianjiao",
        "yezhi",
        "libaojian",
        "nongfushanquan",
        "weitanaiditang",
        "ufo",
        "zihaiguo",
        "nfc",
        "yitengyuan",
        "xianglaniurou",
        "gudasao",
        "buding",
        "ufo2",
        "damaicha",
        "chapai2",
        "tangdaren2",
        "suanlaniurou",
        "bingtangxueli",
        "weitaningmeng-bottle",
        "liziyuan",
        "yousuanru",
        "rancha-1",
        "rancha-2",
        "wanglaoji",
        "weitanai2",
        "qingdaowangzi-1",
        "qingdaowangzi-2",
        "binghongcha",
        "aerbeisi",
        "lujikafei",
        "kele-b-2",
        "anmuxi",
        "xianguolao",
        "haitai",
        "youlemei",
        "weiweidounai",
        "jindian",
        "3jia2",
        "meiniye",
        "rusuanjunqishui",
        "taipingshuda",
        "yida",
        "haochidian",
        "wuhounaicha",
        "baicha",
        "lingdukele-b",
        "jianlibao",
        "lujiaoxiang",
        "3+2-2",
        "luxiangniurou",
        "dongpeng",
        "dongpeng-b",
        "xianxiayuban",
        "niudufen",
        "zaocanmofang",
        "wanglaoji-c",
        "mengniu",
        "mengniuzaocan",
        "guolicheng2",
        "daofandian1",
        "daofandian2",
        "daofandian3",
        "daofandian4",
        "yingyingquqi",
        "lefuqiu",
    )

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        """Same as base detection dataset"""

        super(DetRetailDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, sel_index
        )

        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}

    def load_annotations(self, ann_file):

        data_infos = []
        file_data = file_load(ann_file)
        anno_num = 0
        anno_len = len(file_data["annotations"])
        for img_info in file_data["images"]:
            data_info = {
                "id": img_info["id"],
                "filename": img_info["file_name"],
                "width": img_info["width"],
                "height": img_info["height"],
            }

            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []
            for i in range(anno_num, anno_len):
                if file_data["annotations"][i]["image_id"] == img_info["id"]:
                    label = file_data["annotations"][i]["category_id"]
                    ori_bbox = file_data["annotations"][i]["bbox"]
                    bbox = [
                        ori_bbox[0],
                        ori_bbox[1],
                        ori_bbox[0] + ori_bbox[2],
                        ori_bbox[1] + ori_bbox[3],
                    ]
                    iscrowd = file_data["annotations"][i]["iscrowd"]
                    area = file_data["annotations"][i]["area"]
                    if iscrowd or (area < 10):
                        bboxes_ignore.append(bbox)
                        labels_ignore.append(label)
                    else:
                        bboxes.append(bbox)
                        labels.append(label)
                else:
                    anno_num = i
                    break
            if not bboxes:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0,))
            else:
                bboxes = np.array(bboxes, ndmin=2)
                labels = np.array(labels)
            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 4))
                labels_ignore = np.zeros((0,))
            else:
                bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
                labels_ignore = np.array(labels_ignore)
            data_info["ann"] = dict(
                bboxes=bboxes.astype(np.float32),
                labels=labels.astype(np.int64),
                bboxes_ignore=bboxes_ignore.astype(np.float32),
                labels_ignore=labels_ignore.astype(np.int64),
            )

            data_infos.append(data_info)

        return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""

        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info["width"], img_info["height"]) < min_size:
                continue
            if self.filter_empty_gt and len(img_info["ann"]["bboxes"]) > 0:
                valid_inds.append(i)
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

        return self.data_infos[idx]["ann"]

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]["ann"]["labels"]

    def evaluate(
        self,
        results,
        metric="mAP",
        logger=None,
        proposal_nums=(100, 300, 1000),
        iou_thr=0.5,
        scale_ranges=None,
    ):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
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
            if self.year == 2007:
                ds_name = "voc07"
            else:
                ds_name = self.CLASSES
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset=ds_name,
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
