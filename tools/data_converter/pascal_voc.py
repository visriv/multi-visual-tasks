import argparse
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np

from mvt.utils.data_util import voc_classes
from mvt.utils.misc_util import track_progress
from mvt.utils.io_util import list_from_file, obj_dump
from mvt.utils.path_util import mkdir_or_exist 


label_ids = {name: i for i, name in enumerate(voc_classes())}


def parse_xml(args):
    xml_path, img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        label = label_ids[name]
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(devkit_path, years, split, out_file):
    if not isinstance(years, list):
        years = [years]
    annotations = []
    for year in years:
        filelist = osp.join(devkit_path,
                            f'VOC{year}/ImageSets/Main/{split}.txt')
        if not osp.isfile(filelist):
            print(f'filelist does not exist: {filelist}, '
                  f'skip voc{year} {split}')
            return
        img_names = list_from_file(filelist)
        xml_paths = [
            osp.join(devkit_path, f'VOC{year}/Annotations/{img_name}.xml')
            for img_name in img_names
        ]
        img_paths = [
            f'VOC{year}/JPEGImages/{img_name}.jpg' for img_name in img_names
        ]
        part_annotations = track_progress(parse_xml,
                                          list(zip(xml_paths, img_paths)))
        annotations.extend(part_annotations)
    obj_dump(annotations, out_file)
    return annotations


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to fixed format')
    parser.add_argument('devkit_path', help='pascal voc devkit path')
    parser.add_argument('-o', '--out-dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    mkdir_or_exist(out_dir)

    years = []
    if osp.isdir(osp.join(devkit_path, 'VOC2007')):
        years.append('2007')
    if osp.isdir(osp.join(devkit_path, 'VOC2012')):
        years.append('2012')
    if '2007' in years and '2012' in years:
        years.append(['2007', '2012'])
    if not years:
        raise IOError(f'The devkit path {devkit_path} contains neither '
                      '"VOC2007" nor "VOC2012" subfolder')
    for year in years:
        if year == '2007':
            prefix = 'voc07'
        elif year == '2012':
            prefix = 'voc12'
        elif year == ['2007', '2012']:
            prefix = 'voc0712'
        for split in ['train', 'val', 'trainval']:
            dataset_name = prefix + '_' + split
            print(f'processing {dataset_name} ...')
            cvt_annotations(devkit_path, year, split,
                            osp.join(out_dir, dataset_name + '.pkl'))
        if not isinstance(year, list):
            dataset_name = prefix + '_test'
            print(f'processing {dataset_name} ...')
            cvt_annotations(devkit_path, year, 'test',
                            osp.join(out_dir, dataset_name + '.pkl'))
    print('Done!')


if __name__ == '__main__':
    main()
