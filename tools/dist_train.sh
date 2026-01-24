#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

NNODES=${NUM_NODES:-1}
NODE_RANK=${PBS_NODENUM:-0}
PORT=${MASTER_PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --sync_bn torch \
    --launcher pytorch ${@:3}
