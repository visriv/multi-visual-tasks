import os
import xlrd
import shutil
import traceback
import json
import numpy as np
from lxml.etree import Element, ElementTree, SubElement
from PIL import Image

from mvt.utils.log_util import get_logger


chinese_names = [
    '人物框','卡通人物','游戏角色','猫','狗','蛇','鸟','鱼','兔','猴','马','鸡','猪',
    '牛','羊','自行车','三轮车','摩托车','拖拉机','轿车','巴士','卡车','挖掘机',
    '吊车','火车','飞机','坦克','轮船','别墅','凉亭','塔','寺庙','宫殿','椅子',
    '床','桌子','沙发','长凳','花瓶','盆栽','背包','伞','电脑','电视',
    '台灯','鼠标','键盘','手机','盘子','碗','勺子','瓶子','杯子','叉子','锅',
    '刀','篮球','滑板','书','香蕉','苹果','橙子','西瓜','披萨','蛋糕','汽车','车', 
    '货车', '卡通角色', '凳子', '兔子', '桌椅', '奖杯', '游戏人物'
]

cat2label = {cat: i for i, cat in enumerate(chinese_names)}

english_names = [
    'person', 'cartoon-person', 'game-role', 'cat', 'dog', 'snake',
    'bird', 'fish', 'rabbit', 'monkey', 'horse', 'chicken', 'pig',
    'cow', 'sheep', 'bicycle', 'tricycle', 'motorbike', 'tractor',
    'car', 'bus', 'truck', 'excavator', 'crane', 'train', 'plane',
    'tank', 'ship', 'villa', 'pavilion', 'tower', 'temple', 'palace', 'chair',
    'bed', 'table', 'sofa', 'bench', 'vase', 'potted-plant', 'bag',
    'umbrella', 'computer', 'television', 'lamp', 'mouse', 'keyboard',
    'cell-phone', 'dish', 'bowl', 'spoon', 'bottle', 'cup', 'fork',
    'pot', 'knife', 'basketball', 'skateboard', 'book', 'banana', 'apple',
    'orange', 'watermelon', 'pizza', 'cake','car','car','truck', 
    'cartoon-person', 'bench', 'rabbit', 'table', 'cup', 'game-role']

key_node_names = ['nose', 'lefteye', 'righteye', 'leftear', 'rightear', 
    'leftshoulder', 'rightshoulder','leftelbow', 'rightelbow', 'leftwrist',
    'rightwrist', 'lefthip', 'righthip', 'leftknee', 'rightknee', 
    'leftankle','rightankle', 'headtop']

g_logger = get_logger('error_check', 'meta/test_infos/error_log.txt')
g_error_count = 0
g_all_count = 0


def parse5tags(tag_info_list):
    parse_list_info = []
    if tag_info_list[0].strip()[:2] == '人物':
        parse_list_info.append('person')
    else:
        g_logger.info("Error parsing tag: " + tag_info_list[0])
        return None

    if tag_info_list[1].strip() == '男':
        parse_list_info.append('male')
    elif tag_info_list[1].strip() == '女':
        parse_list_info.append('female')
    else:
        g_logger.info("Error parsing tag: " + tag_info_list[1])
        return None
    
    if tag_info_list[2].strip() == '成人':
        parse_list_info.append('adult')
    elif (tag_info_list[2].strip() == '儿童' or 
            tag_info_list[2].strip() == '小孩' or
            tag_info_list[2].strip() == '孩子'):
        parse_list_info.append('child')
    else:
        g_logger.info("Error parsing tag: " + tag_info_list[2])
        return None

    if tag_info_list[3].strip() == '未截断' or tag_info_list[4].strip() == '未截断':
        parse_list_info.append(0)
    elif tag_info_list[3].strip() == '被截断' or tag_info_list[4].strip() == '被截断':
        parse_list_info.append(1)
    else:
        g_logger.info("Error parsing tag: " + tag_info_list[3])
        return None

    if tag_info_list[4].strip() == '未遮挡' or tag_info_list[3].strip() == '未遮挡':
        parse_list_info.append(0)
    elif tag_info_list[4].strip() == '被遮挡' or tag_info_list[3].strip() == '被遮挡':
        parse_list_info.append(1)
    else:
        g_logger.info("Error parsing tag: " + tag_info_list[4])
        return None
    
    return parse_list_info


def parse3tags(tag_info_list):
    parse_list_info = []
    tag_name = tag_info_list[0].strip()
    if tag_name[-1].isdigit():
        if tag_name[-2:].isdigit():
            tag_name = tag_name[:-2]
        else:
            tag_name = tag_name[:-1]
  
    if tag_name in chinese_names:
        tag_id = cat2label[tag_name]
        parse_list_info.append(english_names[tag_id])
    else:
        g_logger.info('Error tag:' + tag_info_list[0])
        return None

    if tag_info_list[1].strip() == '未截断' or tag_info_list[2].strip() == '未截断':
        parse_list_info.append(0)
    elif tag_info_list[1].strip() == '被截断'  or tag_info_list[2].strip() == '被截断':
        parse_list_info.append(1)
    else:
        g_logger.info('Error tag:' + tag_info_list[1])
        return None

    if tag_info_list[2].strip() == '未遮挡' or tag_info_list[1].strip() == '未遮挡':
        parse_list_info.append(0)
    elif tag_info_list[2].strip() == '被遮挡' or tag_info_list[1].strip() == '被遮挡':
        parse_list_info.append(1)
    else:
        g_logger.info('Error tag:' + tag_info_list[2])
        return None
    
    return parse_list_info


def parse1tags(tag_info_list):
    if tag_info_list[0].strip()[:3] != '人物点':
        g_logger.info("Error parsing tag: " + tag_info_list[0])
        return False
    else:
        return True


def parse_label_content(anno_content, anno_path):
    global g_error_count
    global g_all_count

    person_bbox_list = []
    common_obj_list = []
    person_keynode_list = []

    for object_dict in anno_content['result']['annotation']:
        tag_info_list = object_dict['tag'].split('-')
        g_all_count += 1
        if len(tag_info_list) == 5:
            parsed_info = parse5tags(tag_info_list)

            if parsed_info is None:
                g_logger.info("Error parsing annotation in: " + anno_path + ', ' + object_dict['tag'])
                g_error_count += 1
                continue

            bbox = [
                object_dict['points'][0]['x'],
                object_dict['points'][0]['y'],
                object_dict['points'][2]['x'],
                object_dict['points'][2]['y']
            ]
            person_bbox_list.append({
                'info': parsed_info,
                'bbox': bbox
            })

        elif len(tag_info_list) == 3:
            parsed_info = parse3tags(tag_info_list)

            if parsed_info is None:
                g_logger.info("Error parsing annotation in: " + anno_path + ', ' + object_dict['tag'])
                g_error_count += 1
                continue

            bbox = [
                object_dict['points'][0]['x'],
                object_dict['points'][0]['y'],
                object_dict['points'][2]['x'],
                object_dict['points'][2]['y']
            ]
            common_obj_list.append({
                'info': parsed_info,
                'bbox': bbox
            })
        elif len(tag_info_list) == 1:
            parsed_flag = parse1tags(tag_info_list)
            if not parsed_flag:
                g_error_count += 1
                g_logger.info("Error parsing annotation in: " + anno_path + ', ' + object_dict['tag'])
                continue
            if len(object_dict['points']) != 18:
                g_error_count += 1
                g_logger.info("Error parsing annotation in: " + anno_path + ', ' + object_dict['tag'])
                continue

            person_keynode_list.append(object_dict['points'])

        else:
            g_error_count += 1
            g_logger.info("Error parsing annotation in: " + anno_path + ', ' + object_dict['tag'])

    return person_bbox_list, common_obj_list, person_keynode_list


def get_keynode_bbox(person_keynode_list):

    keynode_bbox_list = []
    for person_keynode in person_keynode_list:
        min_x = 10000
        min_y = 10000
        max_x = 0
        max_y = 0
        
        for i in range(18):
            if 'visible' in person_keynode[i]:
                if person_keynode[i]['visible']:
                    if min_x > person_keynode[i]['x']:
                        min_x = person_keynode[i]['x']
                    if max_x < person_keynode[i]['x']:
                        max_x = person_keynode[i]['x']
                    if min_y > person_keynode[i]['y']:
                        min_y = person_keynode[i]['y']
                    if max_y < person_keynode[i]['y']:
                        max_y = person_keynode[i]['y']
            else:
                if min_x > person_keynode[i]['x']:
                    min_x = person_keynode[i]['x']
                if max_x < person_keynode[i]['x']:
                    max_x = person_keynode[i]['x']
                if min_y > person_keynode[i]['y']:
                    min_y = person_keynode[i]['y']
                if max_y < person_keynode[i]['y']:
                    max_y = person_keynode[i]['y']
        
        keynode_bbox_list.append([min_x, min_y, max_x, max_y])
    return keynode_bbox_list


def get_overlaps(node_bbox, object_bbox_list, eps=1e-3):
    # object_bbox_np = np.array(object_bbox_list)
    area1 = (node_bbox[2] - node_bbox[0]) * (node_bbox[3] - node_bbox[1])
    
    ious = []
    for i in range(len(object_bbox_list)):
        x_start = max(object_bbox_list[i]['bbox'][0], node_bbox[0])
        y_start = max(object_bbox_list[i]['bbox'][1], node_bbox[1])
        x_end = min(object_bbox_list[i]['bbox'][2], node_bbox[2])
        y_end = min(object_bbox_list[i]['bbox'][3], node_bbox[3])
        overlap = max(x_end - x_start, 0) * max(y_end - y_start, 0)
        area2 = (object_bbox_list[i]['bbox'][2] - object_bbox_list[i]['bbox'][0]) * (
            object_bbox_list[i]['bbox'][3] - object_bbox_list[i]['bbox'][1])
    
        union = area2 + area1 - overlap
        union = max(union, eps)
        ious.append(overlap / union)
    
    return ious


def match_keynode_object(keynode_bbox_list, person_bbox_list):
    keynode_ids = []
    # match the most similar bbox
    for keynode_bbox in keynode_bbox_list:
        overlaps = get_overlaps(keynode_bbox, person_bbox_list)
        max_num = np.argmax(overlaps)
        keynode_ids.append(max_num)

    person_keynode_ids = [-1 for i in range(len(person_bbox_list))]
    for i in range(len(keynode_ids)):
        person_keynode_ids[keynode_ids[i]] = i

    return person_keynode_ids


def node_in_bbox(key_node, bbox):
    if (key_node['x'] >= bbox[0] and key_node['x'] <= bbox[2] and
            key_node['y'] >= bbox[1] and key_node['y'] <= bbox[3]):
        return True
    else:
        return False


def xml_generate_test(person_bbox_list, common_obj_list, person_keynode_list,
                      person_keynode_ids, anno_path, width, height):

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'images'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = os.path.basename(anno_path)[:-4] + '.jpg'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    for i in range(len(person_bbox_list)):
        node_object = SubElement(node_root, 'object')

        node_name = SubElement(node_object, 'name')
        node_name.text = person_bbox_list[i]['info'][0]

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = str(person_bbox_list[i]['info'][3])

        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = str(person_bbox_list[i]['info'][4])

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(person_bbox_list[i]['bbox'][0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(person_bbox_list[i]['bbox'][1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(person_bbox_list[i]['bbox'][2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(person_bbox_list[i]['bbox'][3])

        node_truncated = SubElement(node_object, 'gender')
        node_truncated.text = person_bbox_list[i]['info'][1]

        node_difficult = SubElement(node_object, 'age')
        node_difficult.text = person_bbox_list[i]['info'][2]
        
        if person_keynode_ids[i] >= 0:
            node_keynode = SubElement(node_object, 'keynode')
            keynodes = person_keynode_list[person_keynode_ids[i]]
            if node_in_bbox(keynodes[17], person_bbox_list[i]['bbox']):
                key_node_headtop = SubElement(node_keynode, 'headtop')
                key_node_headtop.text = str((keynodes[17]['x'], keynodes[17]['y']))
            for j in range(17):
                if node_in_bbox(keynodes[j], person_bbox_list[i]['bbox']):
                    key_node_headtop = SubElement(node_keynode, key_node_names[j])
                    key_node_headtop.text = str((keynodes[j]['x'], keynodes[j]['y']))
            

    for object_bbox in common_obj_list:
        node_object = SubElement(node_root, 'object')

        node_name = SubElement(node_object, 'name')
        node_name.text = object_bbox['info'][0]

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = str(object_bbox['info'][1])

        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = str(object_bbox['info'][2])

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(object_bbox['bbox'][0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(object_bbox['bbox'][1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(object_bbox['bbox'][2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(object_bbox['bbox'][3])

    # xml_str = tostring(node_root, pretty_print=True)
    tree = ElementTree(node_root)
    tree.write(anno_path, pretty_print=True, xml_declaration=False, encoding='utf-8')


if __name__ == '__main__':
    xlsx_file = '/Users/qianzhiming/Desktop/data/多目标检测标注/PCG/review.xlsx'
    ori_img_dir = '/Users/qianzhiming/Desktop/data/多目标检测标注/PCG/ori_images'
    dst_img_dir = '/Users/qianzhiming/Desktop/data/多目标检测标注/PCG/images'
    dst_anno_dir = '/Users/qianzhiming/Desktop/data/多目标检测标注/PCG/annotations'

    book = xlrd.open_workbook(xlsx_file)

    sheet = book.sheet_by_name('Sheet1')
    
    person_withkn_count = 0
    for i in range(1, sheet.nrows):
        try:
            ori_img_name = sheet.row_values(i)[0] + '.jpg'
            ori_img_path = os.path.join(ori_img_dir, ori_img_name)
            if not os.path.exists(ori_img_path):
                g_logger.info('Path not exists:'+ori_img_path)
                continue
            
            pil_img = Image.open(ori_img_path)

            new_data_name = sheet.row_values(i)[1]
            new_img_path = os.path.join(dst_img_dir, new_data_name + '.jpg')
            shutil.copy(ori_img_path, new_img_path)

            label_content = json.loads(sheet.row_values(i)[2])[0]
            person_bbox_list, common_obj_list, person_keynode_list = parse_label_content(
                label_content, new_data_name)            
            keynode_bbox_list = get_keynode_bbox(person_keynode_list)
            person_withkn_count += len(keynode_bbox_list)
            person_keynode_ids = match_keynode_object(keynode_bbox_list, person_bbox_list)
            new_anno_path = os.path.join(dst_anno_dir, new_data_name + '.xml')
            xml_generate_test(person_bbox_list, common_obj_list, person_keynode_list,
                person_keynode_ids, new_anno_path, pil_img.size[0], pil_img.size[1])
        
        except Exception:
            g_logger.info("---------------------------------------------------------")
            g_logger.info("Exeception information:")
            g_logger.info(traceback.format_exc())
            g_logger.info("Related Path: "+ori_img_name)
            g_logger.info("---------------------------------------------------------")
    
    g_logger.info(f'Number of error annotations: %d' % g_error_count)
    g_logger.info(f'Number of all annotations: %d' % g_all_count)
    g_logger.info(f'Number of person annotations with keynodes: %d' % person_withkn_count)
    g_logger.info(f'Number of object annotations without keynodes: %d' % (g_all_count - person_withkn_count))
