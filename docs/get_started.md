# Get Started

  This page provides basic tutorials about the usage of ObjectiveMTL.
  For installation instructions, please see [install.md](install.md).

  **Important:** please use 'export PYTHONPATH=$(pwd):$PYTHONPATH' to initialize environment

  <!-- TOC -->

  - [Train a Model](#train-a-model)
    - [Train with a single GPU](#train-with-a-single-gpu)
    - [Train with multiple GPUs](#train-with-multiple-gpus)
    - [Export Onnx Models](#export-onnx-models)
  - [Inference with Trained Models](#inference-with-trained-models)
    - [Evaluate a dataset](#evaluate-a-dataset)
    - [Run demos](#run-demos)
    - [Test Onnx Models](#test-onnx-models)
  - [Tutorials](#tutorials)
  - [Task Details](#task-details)

  <!-- TOC -->


## Train a model

  ObjectiveMTL implements distributed training and non-distributed training,
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

  **Important:** When training with the given docker, please refer to 'scripts/mvt_start.sh' 
  Examples:

    ```shell   
    python3 ./tools/train.py --work-dir meta/train_infos --resume-from meta/train_infos/det_ssd_300_vgg_voc/epoch_xxx.pth task_settings/img_det/det_ssd_300_vgg_voc.yaml (ops)
    python3 ./tools/train.py --work-dir meta/train_infos --no-test task_settings/img_det/det_faster_rcnn_r50_fpn_voc.yaml
    python3 ./tools/train.py --work-dir meta/train_infos --no-test task_settings/img_det/det_faster_rcnn_r50_fpn_coco.yaml
    python3 ./tools/train.py --work-dir meta/train_infos --no-test task_settings/img_det/det_ssd_300_vgg_voc.yaml
    python3 ./tools/train.py --work-dir meta/train_infos --no-test task_settings/img_det/det_ssd_300_vgg_coco.yaml
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
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29500  tools/train.py --load-from meta/train_infos/det_yolov3_resnet34_coco/epoch_100.pth --no-test --launcher pytorch task_settings/img_det/det_yolov3_resnet34_coco.yaml
    ```

### Export Onnx Models

    ```shell 
    python3 tools/model_exporter/det_pytorch2onnx.py task_settings/img_det/det_yolov4_cspdarknet_coco.yaml  meta/train_infos/det_yolov4_cspdarknet_coco/epoch_100.pth --input-img meta/test_data/a0519qvbyom_001.jpg --show --output-file meta/onnx_models/det_yolov4_cspdarknet_coco.onnx --opset-version 11 --verify --shape 416 416 --mean 0 0 0 --std 255 255 255
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
  - `--eval`: Evaluation metrics, which depends on the dataset, e.g., "bbox", "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC.
  - `--show`: Whether to show results
  - `--show-dir`: Directory where painted images will be saved.
  - `--show-score-thr`: Score threshold (default: 0.3).
  - `--gpu-collect`: Whether to use gpu to collect results.
  - `--tmpdir`: Tmp directory used for collecting results from multiple workers, available when gpu-collect is not specified.
  - `--launcher`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`.
  - `--local_rank`: ID for local rank. If not specified, it will be set to 0.

  Examples:

    ```shell
    python3 ./tools/test.py task_settings/img_det/det_faster_rcnn_r50_fpn_coco.yaml  meta/train_infos/det_faster_rcnn_r50_fpn_coco/epoch_24.pth --eval 'mAP' --out 'meta/test_infos/det_faster_rcnn_r50_fpn_coco_eval.pkl' --show-dir 'meta/test_infos/det_faster_rcnn_r50_fpn_coco_eval'
    ```

### Run demos

  We also provide scripts to run demos.

  Examples:

    ```shell
    python3 demos/det_infer_demo.py
    python3 demos/det_show_results.py  
    ```

### Test Onnx Models

We also provide scripts to run demos with onnx models.

  Examples:

    ```shell
    python3 demos/det_onnx_test.py
    
    ```


## Tutorials

  Currently, we provide some tutorials for users to:
  - [learn about configs](tutorials/0_config.md)
  - [add a new dataset](tutorials/1_new_dataset.md)
  - [configure data pipelines](tutorials/2_configure_pipeline.md)
  - [add a new model](tutorials/3_new_model.md)
  - [configure training losses](tutorials/4_configure_loss.md)
  - [customize runtime settings](tutorials/5_customize_runtime.md)
  - [other supported operations](tutorials/6_support_detail.md).


## Task Details

  For more details of multiple tasks, please refer to:
  - [detection](tasks/detection.md)
