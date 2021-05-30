# Useful Tools
  This page provides useful tools for data processing.

  <!-- TOC -->

  - [TFRecord Preparation](#tfrecord-preparation)
    - [Generate annotations for the Beauty dataset](#generate-annotations-for-the-beauty-dataset)
    - [Generating tfrecord for the Clearity dataset](#generating-tfrecord-for-the-clearity-dataset)
    - [Generate annotations for the BareDegree dataset](#generate-annotations-for-the-baredegree-dataset)
    - [Generate annotations for the CoverQuality dataset](#generate-annotations-for-the-coverquality-dataset)
    - [Generate annotations for the AnimalDet dataset](#generate-annotations-for-the-animaldet-dataset)
    - [Generate annotations for the CatDogHead dataset](#generate-annotations-for-the-catdogHead-dataset)
    - [Generate annotations for the MultiObjDet dataset](#generate-annotations-for-the-multiobjdet-dataset)
    - [Generate concated images for segmentation](#generate-concated-images-for-segmentation)
  - [Detection visualization and evaluation](#detection-visualization-and-evaluation)
  - [VSCode debugging](#vscode-debugging)

  <!-- TOC -->

## TFRecord Preparation

### Generate annotations for the Beauty dataset
python3 tools/data_converter/anno_regenerate.py
python3 tools/data_converter/tfrecord_test.py --dataset_path '/Users/qianzhiming/Desktop/data/objcls-datasets/Beauty/video' --dataset_type 'cls' --image_dir_name 'images' --split_dir_name 'annotations' --record_path '/Users/qianzhiming/Desktop/data/objcls-datasets/Beauty/video/tfrecords'

### Generating tfrecord for the Clearity dataset
python3 tools/data_converter/tfrecord_test.py --dataset_path '/Users/qianzhiming/Desktop/data/objcls-datasets/Clearity' --dataset_type 'cls' --image_dir_name 'images' --split_dir_name 'annotations' --record_path '/Users/qianzhiming/Desktop/data/objcls-datasets/Clearity/tfrecords'

### Generate annotations for the BareDegree dataset
python3 tools/data_converter/anno_check_format.py
python3 tools/data_converter/tfrecord_test.py --dataset_path '/Users/qianzhiming/Desktop/data/objcls-datasets/BareDegree' --dataset_type 'cls' --image_dir_name 'images' --split_dir_name 'annotations' --record_path '/Users/qianzhiming/Desktop/data/objcls-datasets/BareDegree/tfrecords' --is_file_ext True

### Generate annotations for the CoverQuality dataset
python3 tools/data_converter/anno_check_format.py
python3 tools/data_converter/tfrecord_test.py --dataset_path '/Users/qianzhiming/Desktop/data/objcls-datasets/CoverQuality' --dataset_type 'cls' --image_dir_name 'images' --split_dir_name 'annotations' --record_path '/Users/qianzhiming/Desktop/data/objcls-datasets/CoverQuality/tfrecords' --is_file_ext True

### Generate annotations for the AnimalDet dataset
python3 tools/data_converter/tfrecord_test.py --dataset_path '/Users/qianzhiming/Desktop/data/objdet-datasets/AnimalDet' --dataset_type 'det' --image_dir_name 'images' --label_dir_name 'annotations' --split_dir_name 'metas' --record_path '/Users/qianzhiming/Desktop/data/objdet-datasets/AnimalDet/tfrecords' --label_format 'yolo' --split_str ' '

### Generate annotations for the CatDogHead dataset
python3 tools/data_converter/tfrecord_test.py --dataset_path '/Users/qianzhiming/Desktop/data/objdet-datasets/CatDogHeadDet' --dataset_type 'det' --image_dir_name 'images' --label_dir_name 'annotations' --split_dir_name 'metas' --record_path '/Users/qianzhiming/Desktop/data/objdet-datasets/CatDogHeadDet/tfrecords' --split_str ' '

### Generate annotations for the MultiObjDet dataset
python3 tools/data_converter/multiobj_data_process.py
python3 tools/data_converter/tfrecord_test.py --dataset_path '/Users/qianzhiming/Desktop/data/objdet-datasets/MultiObjDet' --dataset_type 'det' --image_dir_name 'images' --label_dir_name 'annotations' --split_dir_name 'metas' --record_path '/Users/qianzhiming/Desktop/data/objdet-datasets/MultiObjDet/tfrecords' --label_format 'voc'

### Generate concated images for segmentation
python3 tools/data_converter/concat_image_generator.py
python3 tools/data_converter/tfrecord_test.py --dataset_path '/Users/qianzhiming/Desktop/data/other-datasets/ConcatImageSeg' --dataset_type 'seg' --image_dir_name 'images' --label_dir_name 'annotations' --split_dir_name 'metas' --record_path '/Users/qianzhiming/Desktop/data/other-datasets/ConcatImageSeg/tfrecords'


## Detection visualization and evaluation

python3 ./tools/visualization/test_vis_gt_pred.py task_settings/img_det/det_ssd_300_vgg_voc.yaml meta/train_infos/det_ssd_300_vgg_voc/epoch_200.pth --eval 'mAP' --out 'meta/test_infos/det_ssd_300_vgg_voc_eval.pkl' --show-dir 'meta/test_infos/det_ssd_300_vgg_voc_eval'


## VSCode debugging
in the debug console: export PYTHONPATH=$(pwd):$PYTHONPATH

