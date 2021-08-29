import os.path as osp
from PIL import Image
import numpy as np

from .emb_base import EmbBaseDataset
from ..data_builder import DATASETS
from model.mvt.utils.io_util import file_load
from model.mvt.utils.geometric_util import imcrop, imrotate
from model.mvt.utils.bbox_util import get_rotated_bbox


@DATASETS.register_module()
class EmbLocRetailDataset(EmbBaseDataset):
    """Clarity classification"""

    CLASSES = None
    """
        (
        'asamu', 'baishikele', 'baokuangli', 'aoliao', 'bingqilinniunai', 'chapai', 
        'fenda', 'guolicheng', 'haoliyou', 'heweidao', 'hongniu', 'hongniu2', 
        'hongshaoniurou', 'kafei', 'kaomo_gali', 'kaomo_jiaoyan', 'kaomo_shaokao', 
        'kaomo_xiangcon', 'kele', 'laotansuancai', 'liaomian', 'lingdukele', 'maidong', 
        'mangguoxiaolao', 'moliqingcha', 'niunai', 'qinningshui', 'quchenshixiangcao', 
        'rousongbing', 'suanlafen', 'tangdaren', 'wangzainiunai', 'weic', 'weitanai', 
        'weitaningmeng', 'wulongcha', 'xuebi', 'xuebi2', 'yingyangkuaixian', 'yuanqishui', 
        'xuebi-b', 'kebike', 'tangdaren3', 'chacui', 'heweidao2', 'youyanggudong', 
        'baishikele-2', 'heweidao3', 'yibao', 'kele-b', 'AD', 'jianjiao', 'yezhi', 
        'libaojian', 'nongfushanquan', 'weitanaiditang', 'ufo', 'zihaiguo', 'nfc', 
        'yitengyuan', 'xianglaniurou', 'gudasao', 'buding', 'ufo2', 'damaicha', 'chapai2', 
        'tangdaren2', 'suanlaniurou', 'bingtangxueli', 'weitaningmeng-bottle', 'liziyuan', 
        'yousuanru', 'rancha-1', 'rancha-2', 'wanglaoji', 'weitanai2', 'qingdaowangzi-1', 
        'qingdaowangzi-2', 'binghongcha', 'aerbeisi', 'lujikafei', 'kele-b-2', 'anmuxi', 
        'xianguolao', 'haitai', 'youlemei', 'weiweidounai', 'jindian', '3jia2', 'meiniye', 
        'rusuanjunqishui', 'taipingshuda', 'yida', 'haochidian', 'wuhounaicha', 'baicha', 
        'lingdukele-b', 'jianlibao', 'lujiaoxiang', '3+2-2', 'luxiangniurou', 'dongpeng', 
        'dongpeng-b', 'xianxiayuban', 'niudufen', 'zaocanmofang', 'wanglaoji-c', 'mengniu', 
        'mengniuzaocan', 'guolicheng2', 'daofandian1', 'daofandian2', 'daofandian3', 
        'daofandian4', 'yingyingquqi', 'lefuqiu')
    """

    def load_annotations(self):
        """Load data_infos"""

        data_infos = []
        file_data = file_load(self.ann_file)

        img_idx_dict = {}
        for i, img_info in enumerate(file_data["images"]):
            img_idx_dict[img_info["id"]] = i

        for i, bbox_anno in enumerate(file_data["annotations"]):
            img_idx = bbox_anno["image_id"]
            iscrowd = bbox_anno.get("iscrowd", False)
            area = bbox_anno.get("area", 20)  # default value > 10
            if iscrowd or (area < 10):
                continue

            if img_idx not in img_idx_dict:
                continue

            idx = img_idx_dict[img_idx]
            data_infos.append(
                dict(
                    filename=file_data["images"][idx]["file_name"],
                    label=bbox_anno["category_id"],
                    bbox=bbox_anno["bbox"],
                    bbox_id=bbox_anno["id"],
                    width=file_data["images"][idx].get("width", 960),
                    height=file_data["images"][idx].get("height", 720),
                )
            )

        return data_infos

    def prepare_data(self, idx):
        """Prepare data and run pipelines"""

        if self.data_infos[idx]["filename"].endswith(".jpg") or self.data_infos[idx][
            "filename"
        ].endswith(".png"):
            img_path = osp.join(self.data_prefix, self.data_infos[idx]["filename"])
        else:
            img_path = osp.join(
                self.data_prefix, self.data_infos[idx]["filename"] + ".jpg"
            )

        if not osp.exists(img_path):
            raise ValueError(f"Incorrect image path {img_path}.")

        pil_img = Image.open(img_path).convert("RGB")
        img = np.array(pil_img).astype(np.uint8)

        # original bbox
        ori_bbox = [
            self.data_infos[idx]["bbox"][0],
            self.data_infos[idx]["bbox"][1],
            self.data_infos[idx]["bbox"][0] + self.data_infos[idx]["bbox"][2] - 1,
            self.data_infos[idx]["bbox"][1] + self.data_infos[idx]["bbox"][3] - 1,
        ]

        if not self.test_mode:
            # rotate with random angle from [-pi/6, pi/6]
            angle = (np.random.rand() - 0.5) * 60
            img, rot_matrix = imrotate(img, angle, border_value=114, auto_bound=True)
            # print(img.shape[:2])
            # imshow(rot_img[...,[2,1,0]])
            rot_bbox = get_rotated_bbox(
                ori_bbox, rot_matrix, img.shape[1], img.shape[0]
            )
            rot_width = rot_bbox[2] - rot_bbox[0] + 1
            rot_height = rot_bbox[3] - rot_bbox[1] + 1
            x_random_offset = 0.2 * (np.random.rand() - 0.5) * rot_width
            y_random_offset = 0.2 * (np.random.rand() - 0.5) * rot_height
            width_half_diff = int(rot_width * 0.2 * np.random.rand() + 0.5)
            height_half_diff = int(rot_height * 0.2 * np.random.rand() + 0.5)

            crop_bbox = [
                max(rot_bbox[0] + x_random_offset - width_half_diff, 0),
                max(rot_bbox[1] + y_random_offset - height_half_diff, 0),
                min(rot_bbox[2] + x_random_offset + width_half_diff, img.shape[1] - 1),
                min(rot_bbox[3] + y_random_offset + height_half_diff, img.shape[0] - 1),
            ]
        else:
            x_random_offset = 0
            y_random_offset = 0
            width_half_diff = int(self.data_infos[idx]["bbox"][2] * 0.1 + 0.5)
            height_half_diff = int(self.data_infos[idx]["bbox"][2] * 0.1 + 0.5)

            crop_bbox = [
                max(ori_bbox[0] + x_random_offset - width_half_diff, 0),
                max(ori_bbox[1] + y_random_offset - height_half_diff, 0),
                min(ori_bbox[2] + x_random_offset + width_half_diff, img.shape[1] - 1),
                min(ori_bbox[3] + y_random_offset + height_half_diff, img.shape[0] - 1),
            ]

        # crop with bbox
        bbox_img = imcrop(img, np.array(crop_bbox))
        # bbox_img = imcrop(rot_img, rot_bbox)
        # imshow(bbox_img[..., [2, 1, 0]])
        # print(bbox_img.shape[0], bbox_img.shape[1])

        results = {
            "filename": self.data_infos[idx]["filename"],
            "bbox_id": self.data_infos[idx]["bbox_id"],
            "img": bbox_img.astype(np.float32),
            "bbox": [
                float(crop_bbox[0]) / img.shape[1],
                float(crop_bbox[1]) / img.shape[0],
                float(crop_bbox[2]) / img.shape[1],
                float(crop_bbox[3]) / img.shape[0],
            ],
            "label": self.data_infos[idx]["label"],
            "height": bbox_img.shape[0],
            "width": bbox_img.shape[1],
        }

        return self.pipeline(results)
