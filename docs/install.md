## Installation

<!-- TOC -->

- [Requirements](#requirements)
- [Install](#install)
- [Docker Env](#docker-env)
- [Developing](#developing)

<!-- TOC -->

**Important:** Coco install using `pip3 install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"`

### Requirements

- Linux or macOS
- Python 3.6+
- PyTorch 1.5+
- CUDA 9.2+
- GCC 5+
- and so on (seen in requirements)

### Install

pip3 install -r requirements.txt

### Docker Env
docker pull mirrors.tencent.com/tkd_cpc_cv/py36-torch16-mtl:latest

### Developing

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the ObjectiveMTL in the current directory.

To use the default MMPose installed in the environment rather than that you are working with, you can remove the following line in those scripts.

```shell
export PYTHONPATH=$(pwd):$PYTHONPATH
```
or

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
