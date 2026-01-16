#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

source env.sh

NNODES=${NUM_NODES:-1}
# ps -ef|grep python
# hostname
NODE_RANK=${PBS_NODENUM:-0}
PORT=${MASTER_PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# MASTER_ADDR=mg0003
# PORT=29997
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
# export HYDRA_FULL_ERROR=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_TIMEOUT=600

echo "python -m torch.distributed.launch \
    --nnodes=$GPUS \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=1 \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$GPUS \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=1 \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --sync_bn torch \
    --launcher pytorch ${@:3}
