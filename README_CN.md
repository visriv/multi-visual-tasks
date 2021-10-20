<!-- TOC -->
- [介绍](#介绍)
  - [主要特点](#主要特点)
  - [安装](#安装)
  - [开发](#开发)
- [演示](#演示)
  - [训练一个模型](#训练一个模型)
    - [单GPU训练](#单GPU训练)
    - [多GPUs训练](#多GPUs训练)
    - [模型Onnx导出](#模型Onnx导出)
  - [模型推理](#模型推理)
    - [数据集评估](#数据集评估)
    - [测试一个数据集](#测试一个数据集)
    - [评估Embedding](#评估Embedding)
    - [运行测试示例](#运行测试示例)
    - [测试Onnx模型](#测试Onnx模型)
- [常用工具](#常用工具)
  - [加载json文件](#加载json文件)
  - [转换json为xml](#转换json为xml)
  - [打印环境配置信息](#打印环境配置信息)
- [常见问题解答](#常见问题解答)
  - [训练问题](#训练问题)

<!-- TOC -->
# 介绍

  简体中文 | [English](README.md)

  Multi-Visual-Tasks (MVT) 是一个基于Pytorch开发的多视觉任务开源框架。该框架开发了多种现有最先进的视觉方法，例如YoloV4/5, EfficientDet, Swin-Transformer等等。此外，该框架支持2D/3D目标检测、分类、向量特征表示等多种视觉任务。该框架在{MMDetection: https://github.com/open-mmlab/mmdetection}和{Detectron2: https://github.com/facebookresearch/detectron2}等最流行的开源框架基础上进行设计，并不断更新最新视觉领域的科研成果。

## 主要特点

- **支持多种视觉任务**

- **代码简洁且性能高效**

- **支持多种视觉数据集**

- **提供良好地设计接口和开发文档**

## 安装

pip3 install -r requirements.txt

## 开发
**重要说明：** 请使用下列命令进行开发环境初始化

```shell
export PYTHONPATH=$(pwd):$PYTHONPATH
```


# 演示

  该演示包括训练和评估该框架所支持的方法。

## 训练一个模型

  MVT 可以进行分布式和非分布式训练。所有的输出结果（包括日志和模型）都会保存在`work_dir`所指定的工作目录中。缺省状态下每个训练周期完成后都会进行模型评估，但可以根据训练评估设置进行更改。

### 单GPU训练

  ```shell
  python tools/train.py [optional arguments] ${task_yaml_file} 
  ```
  参数说明：

  - `task_yaml_file`: 整体训练设置；
  - `--no-test` (**强烈建议**): 是否在训练过程中进行模型测试；
  - `--work-dir`: 重载设置文件中的工作目录；
  - `--gpus`: GPU使用数量，仅用于非分布式训练；
  - `--gpu-ids`: GPU使用IDs，仅用于非分布式训练；
  - `--seed`: 随机种子生成设置，用于numpy和pytorch；
  - `--deterministic`: 如果指定，则设置CUDNN的deterministic选项；
  - `--launcher`: 分布式训练初始化，可设置为`none`或`pytorch`； 
  - `--local_rank`: 局部rank ID，缺省设置为0；
  - `--opts`: 如果指定，则更改对应的训练设置选项；
  - `--resume-from`：从给定模型开始继续训练，包括训练周期信息；
  - `--load-from`：加载已保存的模型权重，训练周期从0开始，用于模型微调。

  **重要说明:** 训练时可参考bash脚本中的命令，例如'scripts/dist_train.sh'或'scripts/local_train.sh'。
  示例:
    ```shell
    # 检测模型训练
    python3 ./tools/train.py --work-dir meta/train_infos --resume-from meta/train_infos/det_ssd_300_vgg_voc/epoch_xxx.pth tasks/detections/det_ssd_300_vgg_voc.yaml (ops)
    # Embeddinng模型训练
    python3 ./tools/train.py --work-dir meta/train_infos --no-test tasks/embeddings/emb_resnet50_fc_retail.yaml
    ```

### 多GPUs训练

  ```shell
  ./scripts/dist_train.sh ${CONFIG} ${GPU} ${PORT} 
  ```
  或

  ```shell
  python3 -m torch.distributed.launch --nproc_per_node=${GPU} --master_port=${PORT}  tools/train.py --load-from ${CHECKPOINT} --no-test --launcher pytorch ${CONFIG} 
  ```
  示例:

    类似于上诉单卡训练命令，这里给出一些命令行语句。

    ```shell
    # 指定GPUs进行训练
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29500  tools/train.py --load-from meta/train_infos/det_yolov3_resnet34_coco/epoch_xxx.pth --no-test --launcher pytorch tasks/detections/det_yolov3_resnet34_coco.yaml
    ```

### 模型Onnx导出

    ```shell 
    python3 tools/model_exporter/det_pytorch2onnx.py tasks/detections/det_yolov4_cspdarknet_coco.yaml  meta/train_infos/det_yolov4_cspdarknet_coco/epoch_xxx.pth --input-img meta/test_data/xxx.jpg --show --output-file meta/onnx_models/det_yolov4_cspdarknet_coco.onnx --opset-version 11 --verify --shape 416 416 --mean 0 0 0 --std 255 255 255
    ```

## 模型推理

  这里给出测试脚本去评估整个数据集，并提供测试评估的api接口，用于评估扩展。

### 数据集评估

  - [x] 单GPU
  - [x] 单节点多GPUs
  - [x] 多节点

  可以使用以下命令进行数据集评估

  ```shell
  # 单GPU测试
  python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
  ```

  参数说明：

  - `task_config`: 测试配置文件；
  - `checkpoint`: 模型文件；
  - `--out`: 保存输出结果的文件路径；
  - `--fuse-conv-bn`: 是否合并conv和bn，这有利于加速推理速度；
  - `--format-only`: 在不执行评估的前提下格式化输出结果； 
  - `--eval`: 评估指标，与数据集相关，例如COCO中使用的“bbox"和"proposal"和PASCAL VOC使用的"mAP"和"recall"；
  - `--show`: 是否显示输出结果；
  - `--show-dir`: 保存输出结果的文件夹路径；
  - `--show-score-thr`: 置信度阈值(缺省为0.3).
  - `--gpu-collect`: 是否使用GPU来收集结果；
  - `--tmpdir`: 用于从多个工作节点收集结果的临时文件夹路径，仅当gpu-collect没有指定时使用；
  - `--launcher`: 分布式测试的初始化launcher，可设置为`none`或`pytorch`；
  - `--local_rank`: 局部rank ID，如果未指定，则设置为0。

  示例：

    ```shell
    python3 ./tools/test.py tasks/detections/det_faster_rcnn_r50_fpn_coco.yaml  meta/train_infos/det_faster_rcnn_r50_fpn_coco/epoch_24.pth --eval 'mAP' --out 'meta/test_infos/det_faster_rcnn_r50_fpn_coco_eval.pkl' --show-dir 'meta/test_infos/det_faster_rcnn_r50_fpn_coco_eval'
    ```

### 测试一个数据集

  这里提供脚本对dataloader进行测试。

  Examples:

    ```shell
    python3 tools/model_evaluation/eval_with_json_labels.py tasks/detections/det_yolov4_cspdarknet_retail.yaml meta/train_infos/det_yolov4_cspdarknet_retail/epoch_xxx.pth --out-dir meta/test_infos/det_yolov4_cspdarknet_retail --json-path meta/test_infos/a_predictions.json
    ```

### 评估Embedding
    
    该脚本在评估之前保存推理的Embeedings和labels。
    
    示例流程:
    ```shell
    # 在任务配置中进行评估设置
    python3 demos/det_retail_demos/save_embeddings.py tasks/embeddings/emb_resnet50_mlp_retail.yaml meta/train_infos/emb_resnet50_mlp_retail/epoch_100.pth --save-path meta/reference_embedding.pkl

    # 执行评估
    python3 demos/det_retail_demos/eval_embeddings.py meta/reference_embedding.pkl meta/qury_embedding.pkl
    ```

    创建前景目标检测和特征匹配的json文件
    ```shell
    python3 demos/det_retail_demos/eval_with_json_labels.py tasks/detections/det_yolov4_retail_one.yaml \
        meta/train_infos/det_yolov4_retail_one/epoch_xxx.pth \
        --json-path data/test/a_det_annotations.json \
        --out-dir meta/test_a/
    ```
   
    获得预测标签的json文件
    ```shell
    python3 demos/det_retail_demos/pred_embedding_with_json_label.py tasks/embeddings/emb_resnet50_fc_retail.yaml \
        meta/train_infos/emb_resnet50_fc_retail/epoch_xxx.pth \
        meta/reference_test_b_embedding.pkl \
        --json-ori data/test/a_det_annotations.json \
        --json-out submit/out.json
    ```

### 运行测试示例

  模型测试脚本示例:

    ```shell
    python3 demos/det_infer_demo.py 
    ```

### 测试Onnx模型

  Onnx模型测试脚本示例:

    ```shell
    python3 demos/det_onnx_test.py
    
    ```


# 常用工具
  常用工具包括数据处理和日志分析等等。

## 加载json文件

```shell
python3 cell_tests/json_load.py
    
```

## 转换json为xml

```shell
python3 data_converter/json_xml_convert.py
    
```

## 打印环境配置信息

```shell
python3 log_analysis/print_env.py
    
```

# 常见问题解答

  这里罗列了一些常用问题及其解决方案。当遇到问题时，请及时联系和更新问题列表。

## 训练问题

- **TypeError: cannot pickle '_thread._local' object**

  设置WORKERS_PER_DEVICE为0