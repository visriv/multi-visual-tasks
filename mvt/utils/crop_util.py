import math
import numpy as np
import xml.etree.ElementTree as ET

from mtl.utils.io_util import file_load
from mtl.utils.siliency_util import get_saliency_map
from mtl.utils.geometric_util import impad
from mtl.utils.vis_util import imshow_det_bboxes


def merge_boxes(src_box, extra_box):
    """Merge boxes with (minx, miny, maxx, maxy) format"""

    x0 = min(extra_box[0], src_box[0])
    y0 = min(extra_box[1], src_box[1])
    x1 = max(extra_box[2], src_box[2])
    y1 = max(extra_box[3], src_box[3])

    return x0, y0, x1, y1


def validate_occupy_ratio(crop_box, width, height, threshold):
    """Judge the area of the crop box whether satisfy the requirement"""

    if (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1]) / (
            width * height) > threshold:
        return True

    return False


def centered_cmp(box_a, box_b, center_pt):
    # 如果逻辑上认为 a < b ，返回 -1
    # 如果逻辑上认为 a > b , 返回 1
    # 如果逻辑上认为 a == b, 返回 0

    c_x = center_pt[0]
    c_y = center_pt[1]

    c1_x = (box_a[0] + box_a[2]) / 2
    c1_y = (box_a[1] + box_a[3]) / 2
    c2_x = (box_b[0] + box_b[2]) / 2
    c2_y = (box_b[1] + box_b[3]) / 2

    distance1 = abs(c1_x - c_x) + abs(c1_y - c_y)
    distance2 = abs(c2_x - c_x) + abs(c2_y - c_y)

    if distance1 < distance2:
        return -1

    if distance1 > distance2:
        return 1

    return 0


def get_important_box(obj_boxes):
    """Get the most import box from the object boxes"""

    idx = -1
    max_score = 0

    for i, obj_box in enumerate(obj_boxes):
        if obj_box[5] > max_score:
            max_score = obj_box[5]
            idx = i

    if idx > 0:
        return obj_boxes[idx][:4]

    return None


def get_all_ratios(raw_ratio, patch_num):
    large_index = 0 if raw_ratio[0] > raw_ratio[1] else 1
    long_side = raw_ratio[large_index]
    all_ratios = [raw_ratio]
    for i in range(long_side + 1, patch_num + 1):
        new_ratio = [i, i]
        new_ratio[1 - large_index] = int(
            raw_ratio[1 - large_index] * i / long_side + 0.5)
        all_ratios.append(tuple(new_ratio))

    return all_ratios


def resize_img_map(img, sd_map):
    long_side = max(img.shape[0], img.shape[1])
    res_h = long_side - img.shape[0]
    res_w = long_side - img.shape[1]
    pad_h = res_h // 2
    pad_w = res_w // 2

    paddings = (pad_w, pad_h, res_w-pad_w, res_h-pad_h)

    f_sd_map = impad(
        sd_map, 
        padding=paddings, 
        pad_val=0)

    return f_sd_map, paddings


def get_patch_scores(sd_map, patch_num=10):
    assert sd_map.shape[0] == sd_map.shape[1]
    patch_size = sd_map.shape[0] / patch_num

    patch_scores = np.zeros([patch_num, patch_num])
    for j in range(sd_map.shape[0]):
        p_index_j = int(j / patch_size)
        for i in range(sd_map.shape[1]):
            p_index_i = int(i / patch_size)
            patch_scores[p_index_j, p_index_i] += sd_map[j, i]

    patch_scores = patch_scores / (patch_size * patch_size)

    return patch_scores


def get_regions(patch_scores, patch_size, patch_num=10, ratio=(4, 4)):
    assert type(patch_scores) is np.ndarray
    assert patch_scores.shape == (patch_num, patch_num)

    region_props = []
    for j in range(patch_num - ratio[0] + 1):
        for i in range(patch_num - ratio[1] + 1):
            rg_prop = [
                int(i * patch_size),
                int(j * patch_size),
                int((i + ratio[1]) * patch_size) - 1,
                int((j + ratio[0]) * patch_size) - 1]
            rg_score = 0
            for m in range(0, ratio[0]):
                for n in range(0, ratio[1]):
                    rg_score += patch_scores[j + m, i + n]

            rg_score = rg_score / (ratio[0] * ratio[1]) * (math.sqrt(ratio[0] + ratio[1]))

            region_props.append({'rg_bbox': rg_prop, 'rg_score': rg_score})

    return region_props


def topk_regions(rg_props, k):
    sorted_rgs = sorted(
        rg_props, key=lambda x: x['rg_score'], reverse=True)
    return sorted_rgs[:k]


def region_correct(ext_regions, padded_shape, paddings=(0, 0, 0, 0)):

    res_regions = []
    min_x = paddings[0]
    min_y = paddings[1]
    max_x = padded_shape[1] - paddings[2] - 1
    max_y = padded_shape[0] - paddings[3] - 1

    for ext_region in ext_regions:
        bbox_refine = [0, 0, 0, 0]
        if ext_region['rg_bbox'][0] < min_x:
            bbox_refine[0] = 0
        else:
            bbox_refine[0] = ext_region['rg_bbox'][0] - min_x
  
        if ext_region['rg_bbox'][1] < min_y:
            bbox_refine[1] = 0
        else:
            bbox_refine[1] = ext_region['rg_bbox'][1] - min_y

        if ext_region['rg_bbox'][2] > max_x:
            bbox_refine[2] = max_x - min_x
        else:
            bbox_refine[2] = ext_region['rg_bbox'][2] - min_x

        if ext_region['rg_bbox'][3] > max_y:
            bbox_refine[3] = max_y - min_y
        else:
            bbox_refine[3] = ext_region['rg_bbox'][3] - min_y            

        res_regions.append(bbox_refine)

    return res_regions


def get_proposals_from_saliency(img,
                                mask_map=None,
                                img_normalizer=None,
                                partition_ratios=None,
                                patch_num=10,
                                topk_num=3,
                                saliency_type='itti',
                                fixed_size=(320, 320),
                                is_resize=True,
                                draw_img=False,
                                get_saliency=False,
                                color_maps=None):
    """
    get proposals for an images
    """
    assert partition_ratios is not None

    sd_map = get_saliency_map(
        img, saliency_type, img_normalizer, fixed_size, is_resize)

    if mask_map is not None:
        sd_map = sd_map * mask_map

    f_sd_map, paddings = resize_img_map(img, sd_map)

    patch_scores = get_patch_scores(f_sd_map, patch_num)
    patch_size = f_sd_map.shape[0] / patch_num

    final_regions = []
    final_maps = []

    color_count = 0
    padded_shape = (f_sd_map.shape[0], f_sd_map.shape[0])
    for raw_ratio in partition_ratios:
        extended_ratios = get_all_ratios(raw_ratio, patch_num)
        rg_props = []
        for ratio in extended_ratios:
            rg_props.extend(
                get_regions(patch_scores, patch_size, patch_num, ratio))

        sorted_rg_props = topk_regions(rg_props, topk_num)
        
        # remove paddings from results
        ext_regions = region_correct(sorted_rg_props, padded_shape, paddings)

        final_regions.append(ext_regions)
        if color_maps is not None:
            final_maps.append(color_maps[color_count])
        color_count += 1

    results = [final_regions, final_maps]

    if draw_img:
        color_count = 0
        for rg_props in final_regions:
            label_list = []
            for _ in range(len(rg_props)):
                label_list.append(0)

            bbox_np = np.array(rg_props)
            # print(bbox_np)
            label_np = np.array(label_list)
            imshow_det_bboxes(
                img, bbox_np, label_np,
                class_names=['prop'],
                bbox_color=final_maps[color_count],
                show=False)
            color_count += 1
        results.append(img)

    if get_saliency:
        results.append(sd_map)

    return tuple(results)


def get_crops(json_anno_path):
    try:
        anno_data = file_load(json_anno_path)
    except Exception:
        print(json_anno_path)
        raise ValueError('Unexpected labels')

    anno_data.pop('imageData')

    crop_bbox_list = []

    for item in anno_data['shapes']:
        # process with each object

        bbox_xyxy = [
            float(item['points'][0][0]),
            float(item['points'][0][1]),
            float(item['points'][1][0]),
            float(item['points'][1][1])
        ]
        crop_bbox_list.append(bbox_xyxy)

    return crop_bbox_list


def get_object_labels(xml_anno_path, cat2obj_label, key_node_names):

    tree = ET.parse(xml_anno_path)
    root = tree.getroot()
    bbox_list = []
    label_list = []
    keynode_list = []
    img_width = int(root.find('size').find('width').text)
    img_height = int(root.find('size').find('height').text)

    for obj in root.findall('object'):

        bnd_box = obj.find('bndbox')
        min_x = int(float(bnd_box.find('xmin').text))
        min_y = int(float(bnd_box.find('ymin').text))
        max_x = int(float(bnd_box.find('xmax').text))
        max_y = int(float(bnd_box.find('ymax').text))
        if max_x < min_x + 3 or max_y < min_y + 3:
            continue

        # process with each object
        name = obj.find('name').text
        label = cat2obj_label[name]
        label_list.append(label)
        bbox = [min_x, min_y, max_x, max_y]
        bbox_list.append(bbox)

        if name == 'person':
            key_node_handle = obj.find('keynode')
            key_nodes = {}

            # the key node may be lost, we just keep the labelled node
            if key_node_handle is not None:
                for i in range(len(key_node_names)):
                    if key_node_handle.find(key_node_names[i]) is not None:
                        key_nodes[key_node_names[i]] = list(eval(
                            key_node_handle.find(key_node_names[i]).text))
            
                keynode_list.append(key_nodes)
        else:
            keynode_list.append({})
        
    return bbox_list, label_list, keynode_list, img_width, img_height


def get_face_ocr_edge_labels(json_anno_path):
    anno_data = file_load(json_anno_path)
    bbox_list = []
    label_list = []
    face_node_list = []
    
    if 'data' in anno_data['human_face']:
        if len(anno_data['human_face']['data']['resultList']) > 1:
            raise ValueError('Result list error')
        for face_item in anno_data['human_face']['data']['resultList'][0]['faceList']:
            # process with each object
            label_list.append(65)
            try:
                min_x = 0
                if 'x' in face_item['pos']['min']:
                    min_x = int(float(face_item['pos']['min']['x']))
                min_y = 0
                if 'y' in face_item['pos']['min']:
                    min_y = int(float(face_item['pos']['min']['y']))
                bbox = [
                    min_x,
                    min_y,
                    int(float(face_item['pos']['max']['x'])),
                    int(float(face_item['pos']['max']['y']))
                ]
            except:
                print(json_anno_path)
                raise ValueError('Face coordinates error!')

            bbox_list.append(bbox)
            try:
                face_node_list.append(
                    {
                        'lefteye': [float(face_item['lm']['points'][0]['x']),
                                    float(face_item['lm']['points'][0]['y'])],
                        'righteye': [float(face_item['lm']['points'][1]['x']),
                                     float(face_item['lm']['points'][1]['y'])],
                        'nose': [float(face_item['lm']['points'][2]['x']),
                                 float(face_item['lm']['points'][2]['y'])]
                    }
                )
            except Exception:
                print(json_anno_path)
                continue
                # raise ValueError('Face landmark error!')

    for ocr_item in anno_data['ocr_boxes']:
        label_list.append(69)
        bbox_list.append(ocr_item[1])
        face_node_list.append({})

    if 'edge' in anno_data:
        label_list.append(68)
        edge_hit = eval(anno_data['edge']['cc_ai_content_edge_detection_hit_result'])
        edge_info = eval(anno_data['edge']['cc_ai_content_edge_detection_result'])
        if edge_hit > 0:
            bbox_list.append(edge_info['huabian'])
        face_node_list.append({})

    return bbox_list, label_list, face_node_list


def get_cat_dog_labels(xml_anno_path):
    tree = ET.parse(xml_anno_path)
    root = tree.getroot()
    bbox_list = []
    label_list = []

    for obj in root.findall('object'):
        # process with each object
        name = obj.find('name').text
        if name == 'cat':
            label_list.append(66)
        else:
            label_list.append(67)

        bnd_box = obj.find('bndbox')
        bbox = [
            int(float(bnd_box.find('xmin').text)),
            int(float(bnd_box.find('ymin').text)),
            int(float(bnd_box.find('xmax').text)),
            int(float(bnd_box.find('ymax').text))]
        bbox_list.append(bbox)

    return bbox_list, label_list


def get_area(bbox):
    return (bbox[2] - bbox[0])*(bbox[3] - bbox[1])


def is_in_region(pt, region_bbox):
    if pt[0] < region_bbox[0] or pt[0] > region_bbox[2]:
        return False
    elif pt[1] < region_bbox[1] or pt[1] > region_bbox[3]:
        return False
    else:
        return True


def find_area_index(area, rel_size_splits):
    for idx, rel_size_split in enumerate(rel_size_splits):
        if area <= rel_size_split:
            return idx
    return len(rel_size_splits) - 1


def find_iou_index(iou, conflict_ious):
    for idx, conflict_iou in enumerate(conflict_ious):
        if iou <= conflict_iou:
            return idx
    return len(conflict_ious) - 1


def calculate_iou(bbox_1, bbox_2):

    min_x = max(bbox_1[0], bbox_2[0])
    max_x = min(bbox_1[2], bbox_2[2])
    if min_x >= max_x:
        return 0
    min_y = max(bbox_1[1], bbox_2[1])
    max_y = min(bbox_1[3], bbox_2[3])
    if min_y >= max_y:
        return 0

    area = get_area(bbox_2)

    return (max_x - min_x) * (max_y - min_y) / area


def calculate_conflict_block(bbox_1, bbox_2):
    conflicts_block = np.zeros((9))

    w_2 = bbox_2[2] - bbox_1[0]

    if bbox_1[0] < bbox_2[0]:
        xmin = 0
    elif bbox_1[0] < bbox_2[0] + w_2/3:
        xmin = 1
    elif bbox_1[0] < bbox_2[0] + w_2*2/3:
        xmin = 2
    else:
        xmin = 3

    if bbox_1[2] < bbox_2[0] + w_2/3:
        xmax = 1
    elif bbox_1[2] < bbox_2[0] + w_2*2/3:
        xmax = 2
    elif bbox_1[2] < bbox_2[0] + w_2:
        xmax = 3
    else:
        xmax = 4

    if bbox_1[1] < bbox_2[1]:
        ymin = 0
    elif bbox_1[1] < bbox_2[1] + w_2/3:
        ymin = 1
    elif bbox_1[1] < bbox_2[1] + w_2*2/3:
        ymin = 2
    else:
        ymin = 3

    if bbox_1[3] < bbox_2[1] + w_2/3:
        ymax = 1
    elif bbox_1[3] < bbox_2[1] + w_2*2/3:
        ymax = 2
    elif bbox_1[3] < bbox_2[1] + w_2:
        ymax = 3
    else:
        ymax = 4

    for i in range(xmin, xmax):
        if i == 0:
            continue
        if ymin >=1 and ymin <= 3:
            conflicts_block[(ymin - 1)*3 + i-1] = 1
        if ymax >=1 and ymax <= 3:
            conflicts_block[(ymax - 1)*3 + i-1] = 1

    for i in range(ymin, ymax):
        if i == 0:
            continue
        if xmin >=1 and xmin <= 3:
            conflicts_block[(i - 1)*3 + xmin - 1] = 1
        if xmax >=1 and xmax <= 3:
            conflicts_block[(i - 1)*3 + xmax - 1] = 1

    return conflicts_block
