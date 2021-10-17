#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
METRIC=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python3 tools/test.py $CONFIG \
    $CHECKPOINT --eval $METRIC
