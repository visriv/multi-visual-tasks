#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
GPU_IDS=
SPLIT=", "

for i in $(seq 0 $GPUS):
do
   GPU_IDS=${GPU_IDS}${i}${SPLIT};
done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py --no-test --launcher pytorch $CONFIG ${@:3}

