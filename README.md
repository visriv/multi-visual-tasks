## Introduction

  MultipleVisualTasks (MVT) is an open-source toolbox for multiple vision tasks based on PyTorch, following the most advanced algorithm. Our repository is designed according to {MMDetection: https://github.com/open-mmlab/mmdetection} and {Detectron2: https://github.com/facebookresearch/detectron2}.

### Major Features

- **Support multiple tasks**

  Currently, We only support the vision task of image detection and embedding. 

- **Well designed, tested and documented**

  We decompose MVT into different components and one can easily construct a customized vision task by combining different modules.
  We provide detailed documentation and API reference.


## Installation

### Requirements

- Linux or macOS
- Python 3.6+
- PyTorch 1.5+
- CUDA 9.2+
- GCC 5+
- and so on (seen in requirements)

### Install

pip3 install -r requirements.txt

### Developing

The train and test scripts already modify the `PYTHONPATH` to ensure the script in the current directory.

```shell
export PYTHONPATH=$(pwd):$PYTHONPATH
```
or

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## Get Started

### Train a model with a single GPU

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
    python3 ./tools/train.py --work-dir model_files/ --no-test model/task_settings/img_det/det_yolov4_9a_retail_one.yaml
    python3 ./tools/train.py --work-dir model_files/ --no-test model/task_settings/img_emb/emb_resnet50_mlp_loc_retail.yaml
    ```

### Train with multiple GPUs

  ```shell
  ./scripts/dist_train.sh ${CONFIG} ${GPU} ${PORT} 
  ```
  or

  ```shell
  python3 -m torch.distributed.launch --nproc_per_node=${GPU} --master_port=${PORT}  tools/train.py --work-dir ${WORK_DIR} --load-from ${CHECKPOINT} --no-test --launcher pytorch ${CONFIG} 
  ```
  Examples:

    Similar to training cases with a single GPU, here gives a few examples.

    ```shell
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29500  tools/train.py --work-dir model_files/ --no-test --launcher pytorch model/task_settings/img_det/det_yolov4_9a_retail_one.yaml
    ```

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
    python3 ./tools/test.py task_settings/img_det/det_yolov4_9a_retail_one.yaml  model_files/det_yolov4_9a_retail_one/epoch_200.pth --eval 'mAP' --out 'model_files/det_yolov4_9a_retail_one_eval.pkl' --show-dir 'data/det_yolov4_9a_retail_one_eval'
    ```

### Test a dataset

  We also provide scripts to test with a dataloader.

  Examples:

    ```shell
    python3 tools/model_evaluation/eval_with_json_labels.py task_settings/img_det/det_yolov4_cspdarknet_retail.yaml meta/train_infos/det_yolov4_cspdarknet_retail/epoch_100.pth --out-dir meta/test_infos/det_yolov4_cspdarknet_retail --json-path meta_test_infos/a_predictions.json
    ```

### Evaluate embedding
    
    Save reference embeddings and labels before evaluation:
    
    Examples:
    ```shell
    # set validation set by reference set in task config
    python3 tools/model_evaluation/save_embeddings.py task_settings/img_emb/emb_resnet50_fc_retail.yaml meta/train_infos/emb_resnet50_fc_retail/epoch_500.pth --save-path meta/reference_embedding.pkl
    # set validation set by query set in task config
    python3 tools/model_evaluation/save_embeddings.py task_settings/img_emb/emb_resnet50_fc_retail.yaml meta/train_infos/emb_resnet50_fc_retail/epoch_500.pth --save-path meta/qury_embedding.pkl
    # run evaluation
    python3 tools/model_evaluation/eval_embeddings.py meta/reference_embedding.pkl meta/qury_embedding.pkl
    ```

### Create retail submision file
    Create detection json file:
    ```shell
    python3 tools/model_evaluation/eval_with_json_labels.py task_settings/img_det/det_yolov4_retail_one.yaml \
        meta/train_infos/det_yolov4_retail_one/epoch_200.pth \
        --json-path data/test/a_det_annotations.json \
        --out-dir meta/test_a/
    ```
   
    Get predicted labels and save final submition json file:
    ```shell
    python3 tools/model_evaluation/pred_embedding_with_json_label.py task_settings/img_emb/emb_resnet50_fc_retail.yaml \
        meta/train_infos/emb_resnet50_fc_retail/epoch_50.pth \
        meta/reference_test_b_embedding.pkl \
        --json-ori data/test/a_det_annotations.json \
        --json-out submit/out.json
    python3 tools/model_evaluation/pred_embedding_with_json_label.py task_settings/img_emb/emb_resnet50_mlp_loc_retail.yaml meta/train_infos/emb_resnet50_mlp_loc_retail/epoch_41.pth meta/reference_test_b_embedding.pkl --json-ori data/RetailDet/test/a_det_annotations.json --json-out submit/out.json
    ```

### Run demos

  We also provide scripts to run demos.

  Examples:

    ```shell
    python3 demos/det_infer_demo.py 
    ```
