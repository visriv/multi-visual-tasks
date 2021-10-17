#!/usr/bin/env bash

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 tools/train.py --no-test ${@:2} $CONFIG
