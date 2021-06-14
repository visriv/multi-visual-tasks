## Introduction

  English | [简体中文](README_CN.md)

  MultipleVisualTasks (MVT) is an open-source toolbox for multiple vision tasks based on PyTorch, following the most advanced algorithm, such as YoloV4/5, EfficientDet, Swin-T, and so on. Our repository is designed according to {MMDetection: https://github.com/open-mmlab/mmdetection} and {Detectron2: https://github.com/facebookresearch/detectron2}.

### Major Features

- **Support multiple tasks**

  Currently, We only support the vision task of image detection, however, the interfaces of datasets and models are open to other tasks, which will be open lately.

- **Higher efficiency and higher accuracy**

  MVT implements multiple state-of-the-art (SOTA) vision models. 

- **Support for various datasets**

  The toolbox directly supports multiple representative and custom datasets.

- **Well designed, tested and documented**

  We decompose MVT into different components and one can easily construct a customized vision task by combining different modules.
  We provide detailed documentation and API reference.


## Installation

  Please refer to [install.md](docs/install.md) for installation.


## Get Started

  Please see [get_started.md](docs/get_started.md) for the basic usage of ObjectiveMTL.

  There are also tutorials:
  - [learn about configs](docs/tutorials/0_config.md)
  - [add a new dataset](docs/tutorials/1_new_dataset.md)
  - [configure data pipelines](docs/tutorials/2_configure_pipeline.md)
  - [add a new model](docs/tutorials/3_new_model.md)
  - [configure training losses](docs/tutorials/4_configure_loss.md)
  - [customize runtime settings](docs/tutorials/5_customize_runtime.md)
  - [other supported operations](docs/tutorials/6_support_detail.md)


## Useful Tools

  Please refer to [useful_tools.md](docs/useful_tools.md) for auxiliary tests.


## FAQ

  Please refer to [FAQ](docs/faq.md) for frequently asked questions.
