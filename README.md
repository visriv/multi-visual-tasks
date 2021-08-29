<!-- TOC -->
- [Introduction](#introduction)
  - [Major Features](#major-features)
  - [Installation](#installation)
  - [Developing](#developing)
- [Get Started](#get-started)
  - [Train a Model](#train-a-model)
    - [Train with a single GPU](#train-with-a-single-gpu)
    - [Train with multiple GPUs](#train-with-multiple-gpus)
    - [Export Onnx Models](#export-onnx-models)
  - [Inference with Trained Models](#inference-with-trained-models)
    - [Evaluate a dataset](#evaluate-a-dataset)
    - [Test a dataset](#test-a-dataset)
    - [Run demos](#run-demos)
    - [Test Onnx Models](#test-onnx-models)
  - [Tutorials](#tutorials)
  - [Task Details](#task-details)
- [useful Tools](#useful-tools)
  - [Loading json file](#loading-json-file)
  - [Convert json to xml](#convert-json-to-xml)
  - [Print environment](#print-environment)
- [FAQ](#faq)
  - [Training Questions](#training-questions)

<!-- TOC -->
# Introduction

  English | [简体中文](README_CN.md)

  Multi-Visual-Tasks (MVT) is an open-source toolbox for multiple vision tasks based on PyTorch, following the most advanced algorithms, such as YoloV4/5, EfficientDet, Swin-Transformer, and so on. Our repository is designed according to {MMDetection: https://github.com/open-mmlab/mmdetection} and {Detectron2: https://github.com/facebookresearch/detectron2}.

## Major Features

- **Support multiple tasks with different configs**

- **Higher efficiency and higher accuracy**

- **Support for various datasets**

- **Well designed, tested and documented**

## Installation

pip3 install -r requirements.txt

## Developing

**Important:** please use the following command to initialize environment

```shell
export PYTHONPATH=$(pwd):$PYTHONPATH
```


# Get Started

  We start our repository by training and evaluating with the supported methods.

## Train a model

  MVT implements distributed training and non-distributed training,
  which uses `CustomDistributedDataParallel` and `CustomDataParallel` respectively.

  All outputs (log files and checkpoints) will be saved to the working directory,
  which is specified by `work_dir` in the config file.

  By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by modifying the interval argument in the training config

### Train with a single GPU

  ```shell
  python tools/train.py [optional arguments] ${task_yaml_file} 
  ```
  Optional arguments are:

  - `task_yaml_file`: train setting file for configuration.
  - `--no-test` (**strongly recommended**): whether not to evaluate the checkpoint during training.
  - `--work-dir`: Override the working directory specified in the config file.
  - `--gpus`: Number of gpus to use, which is only applicable to non-distributed training.
  - `--gpu-ids`: ids of gpus to use, only applicable to non-distributed training
  - `--seed`: Seed id for random state in python, numpy and pytorch to generate random numbers.
  - `--deterministic`: If specified, it will set deterministic options for CUDNN backend.
  - `--launcher`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`. 
  - `--local_rank`: ID for local rank. If not specified, it will be set to 0.
  - `--opts`: If specified, it will modify config options using the command-line.
  - `--resume-from` the checkpoint file to resume from, including epoch information.
  - `--load-from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

  **Important:** When training with the given docker, please refer to 'scripts/dist_train.sh' or 'scripts/local_train.sh'
  Examples:
    ```shell   
    python3 ./tools/train.py --work-dir meta/train_infos --resume-from meta/train_infos/det_ssd_300_vgg_voc/epoch_xxx.pth tasks/detections/det_ssd_300_vgg_voc.yaml (ops)
    python3 ./tools/train.py --work-dir meta/train_infos --no-test tasks/embeddings/emb_resnet50_fc_retail.yaml
    ```

### Train with multiple GPUs

  ```shell
  ./scripts/dist_train.sh ${CONFIG} ${GPU} ${PORT} 
  ```
  or

  ```shell
  python3 -m torch.distributed.launch --nproc_per_node=${GPU} --master_port=${PORT}  tools/train.py --load-from ${CHECKPOINT} --no-test --launcher pytorch ${CONFIG} 
  ```
  Examples:

    Similar to training cases with a single GPU, here gives a few examples.

    ```shell
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29500  tools/train.py --load-from meta/train_infos/det_yolov3_resnet34_coco/epoch_xxx.pth --no-test --launcher pytorch tasks/detections/det_yolov3_resnet34_coco.yaml
    ```

### Export Onnx Models

    ```shell 
    python3 tools/model_exporter/det_pytorch2onnx.py tasks/detections/det_yolov4_cspdarknet_coco.yaml  meta/train_infos/det_yolov4_cspdarknet_coco/epoch_xxx.pth --input-img meta/test_data/xxx.jpg --show --output-file meta/onnx_models/det_yolov4_cspdarknet_coco.onnx --opset-version 11 --verify --shape 416 416 --mean 0 0 0 --std 255 255 255
    ```

## Inference with Trained Models

  We provide testing scripts to evaluate a whole dataset, and provide some high-level apis for easier integration to other projects.

### Evaluate a dataset

  - [x] single GPU
  - [x] single node multiple GPUs
  - [x] multiple node

  You can use the following commands to test a dataset.

  ```shell
  # single-gpu testing
  python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
  ```

  Optional arguments:

  - `task_config`: Test config file.
  - `checkpoint`: Checkpoint file.
  - `--out`: Filename of the output results. If not specified, the results will not be saved to a file.
  - `--fuse-conv-bn`: Whether to fuse conv and bn, this will slightly increase the inference speed
  - `--format-only`: Format the output results without perform evaluation. 
  - `--eval`: Evaluation metrics, which depends on the dataset, e.g., "bbox", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC.
  - `--show`: Whether to show results
  - `--show-dir`: Directory where painted images will be saved.
  - `--show-score-thr`: Score threshold (default: 0.3).
  - `--gpu-collect`: Whether to use gpu to collect results.
  - `--tmpdir`: Tmp directory used for collecting results from multiple workers, available when gpu-collect is not specified.
  - `--launcher`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`.
  - `--local_rank`: ID for local rank. If not specified, it will be set to 0.

  Examples:

    ```shell
    python3 ./tools/test.py tasks/detections/det_faster_rcnn_r50_fpn_coco.yaml  meta/train_infos/det_faster_rcnn_r50_fpn_coco/epoch_24.pth --eval 'mAP' --out 'meta/test_infos/det_faster_rcnn_r50_fpn_coco_eval.pkl' --show-dir 'meta/test_infos/det_faster_rcnn_r50_fpn_coco_eval'
    ```

### Test a dataset

  We also provide scripts to test with a dataloader.

  Examples:

    ```shell
    python3 tools/model_evaluation/eval_with_json_labels.py tasks/detections/det_yolov4_cspdarknet_retail.yaml meta/train_infos/det_yolov4_cspdarknet_retail/epoch_xxx.pth --out-dir meta/test_infos/det_yolov4_cspdarknet_retail --json-path meta/test_infos/a_predictions.json
    ```

### Evaluate embedding
    
    Save reference embeddings and labels before evaluation:
    
    Examples:
    ```shell
    # set validation set by reference set in task config
    python3 tools/model_evaluation/save_embeddings.py tasks/embeddings/emb_resnet50_fc_retail.yaml meta/train_infos/emb_resnet50_fc_retail/epoch_500.pth --save-path meta/reference_embedding.pkl

    # set validation set by query set in task config
    python3 tools/model_evaluation/save_embeddings.py tasks/embeddings/emb_resnet50_fc_retail.yaml meta/train_infos/emb_resnet50_fc_retail/epoch_500.pth --save-path meta/qury_embedding.pkl

    # run evaluation
    python3 tools/model_evaluation/eval_embeddings.py meta/reference_embedding.pkl meta/qury_embedding.pkl
    ```

### Create the json file for various object detection by positive object detection and embedding match
    Create detection json file:
    ```shell
    python3 tools/model_evaluation/eval_with_json_labels.py tasks/detections/det_yolov4_retail_one.yaml \
        meta/train_infos/det_yolov4_retail_one/epoch_xxx.pth \
        --json-path data/test/a_det_annotations.json \
        --out-dir meta/test_a/
    ```
   
    Get predicted labels and save final submition json file:
    ```shell
    python3 tools/model_evaluation/pred_embedding_with_json_label.py tasks/embeddings/emb_resnet50_fc_retail.yaml \
        meta/train_infos/emb_resnet50_fc_retail/epoch_xxx.pth \
        meta/reference_test_b_embedding.pkl \
        --json-ori data/test/a_det_annotations.json \
        --json-out submit/out.json

    python3 tools/model_evaluation/pred_embedding_with_json_label.py tasks/embeddings/emb_resnet50_mlp_loc_retail.yaml meta/train_infos/emb_resnet50_mlp_loc_retail/epoch_xxx.pth meta/reference_test_b_embedding.pkl --json-ori data/RetailDet/test/a_det_annotations.json --json-out meta/test/output.json
    ```

### Run demos

  We also provide scripts to run demos.

  Examples:

    ```shell
    python3 demos/det_infer_demo.py 
    ```

### Test Onnx Models

We also provide scripts to run demos with onnx models.

  Examples:

    ```shell
    python3 demos/det_onnx_test.py
    
    ```


# Useful Tools
  We give some tools for data process, log analysis, and so on.

## Loading json file

```shell
python3 cell_tests/json_load.py
    
```

## Convert json to xml

```shell
python3 data_converter/json_xml_convert.py
    
```

## Print environment

```shell
python3 log_analysis/print_env.py
    
```

# FAQ

  We list some common issues faced by many users and their corresponding solutions here.
Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them.

## Training Questions

- **TypeError: cannot pickle '_thread._local' object**

  Set WORKERS_PER_DEVICE as 0