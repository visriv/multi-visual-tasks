import os.path as osp
from PIL import Image
import numpy as np

from .emb_base import EmbBaseDataset
from ..data_builder import DATASETS
from mvt.utils.io_util import file_load
from mvt.utils.geometric_util import imcrop


@DATASETS.register_module()
class EmbRetailDataset(EmbBaseDataset):
    """Clarity classification"""

    CLASSES = (
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

    def load_annotations(self):
        """Load data_infos"""

        data_infos = []
        file_data = file_load(self.ann_file)
	
        for i, img_info in enumerate(file_data['images']):
            assert img_info['id'] == i

        for i, bbox_anno in enumerate(file_data['annotations']):
            img_idx = bbox_anno['image_id']
            iscrowd = bbox_anno['iscrowd']
            area = bbox_anno['area']
            if iscrowd or (area < 10):
                continue

            data_infos.append(dict(
                filename=file_data['images'][img_idx]['file_name'],
                label=bbox_anno['category_id'],
		        bbox=bbox_anno['bbox'],
                width=file_data['images'][img_idx]['width'],
                height=file_data['images'][img_idx]['height']))
        
        return data_infos

    
    def prepare_data(self, idx):
        """Prepare data and run pipelines"""
        
        if self.data_infos[idx]['filename'].endswith('.jpg') or self.data_infos[idx]['filename'].endswith('.png'):
            img_path = osp.join(
                self.data_prefix, self.data_infos[idx]['filename'])
        else:
            img_path = osp.join(
                self.data_prefix, self.data_infos[idx]['filename'] + '.jpg')

        if not osp.exists(img_path):
            raise ValueError(f"Incorrect image path {img_path}.")

        pil_img = Image.open(img_path).convert('RGB')
        img = np.array(pil_img).astype(np.uint8)

        # original bbox
        ori_bbox = np.array(self.data_infos[idx]['bbox'])
        x_random_offset = 0.2 * (np.random.rand() - 0.5) * ori_bbox[2]
        y_random_offset = 0.2 * (np.random.rand() - 0.5) * ori_bbox[3]
        width_ratio = 1 - 0.2 * (np.random.rand() - 0.5)
        height_ratio = 1 - 0.2 * (np.random.rand() - 0.5)


        crop_bbox = [
            min(ori_bbox[0] + x_random_offset, 0),
            min(ori_bbox[1] + y_random_offset, 0),
            max(ori_bbox[0] + int(ori_bbox[2] * width_ratio) - 1, self.data_infos[idx]['width'] - 1),
            max(ori_bbox[1] + int(ori_bbox[2] * height_ratio) - 1, self.data_infos[idx]['height'] - 1)
        ]

        # crop with bbox	
        bbox_img = imcrop(img, np.array(crop_bbox))

        results = {
            'filename': self.data_infos[idx]['filename'],
            'img': bbox_img.astype(np.float32),
            'label': self.data_infos[idx]['label'],
            'height': bbox_img.shape[0],
            'width': bbox_img.shape[1]
            }

        return self.pipeline(results)
