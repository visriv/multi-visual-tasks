## Introduction

  English | [简体中文](README_CN.md)

  Introduction: https://docs.qq.com/doc/DTXhFY2NWZm9NWURo
  Develop API: https://docs.qq.com/doc/DTU5IdFljcXRkVWVC

  ObjectiveMTL is an open-source toolbox for multiple vision tasks based on PyTorch.

### Major Features

- **Support multiple tasks**

  We support a wide spectrum of vision tasks in current research community, including image classification, image detection and instance segmentation, image segmentation, 
  data regression and pose estimation.

- **Higher efficiency and higher accuracy**

  MTL implements multiple state-of-the-art (SOTA) vision models. 

- **Support for various datasets**

  The toolbox directly supports multiple representative and custom datasets.

- **Well designed, tested and documented**

  We decompose ObjectiveMTL into different components and one can easily construct a customized vision task by combining different modules.
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


## Supported algorithms

  Algorithm details and trained models with corresponding evaluations:
  - [classification](docs/tasks/classification.md)
  - [detection](docs/tasks/detection.md)
  - [pose estimation](docs/tasks/pose_estimation.md)
  - [regression](docs/tasks/regression.md)
  - [segmentation](docs/tasks/segmentation.md)


## Useful Tools

  Please refer to [useful_tools.md](docs/useful_tools.md) for auxiliary tests.


## FAQ

  Please refer to [FAQ](docs/faq.md) for frequently asked questions.
