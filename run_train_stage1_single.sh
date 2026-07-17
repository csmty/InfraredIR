#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

accelerate launch --num_processes=1 --gpu_ids="1," --main_process_port 29300 src/train.py \
    --sd_path="/data/matengyu/code/@submit/Infrared/CVPR26/S3Diff/weight/sd-turbo/models--stabilityai--sd-turbo/snapshots/sd-turbo" \
    --base_config="configs/train_STAGE1_single.yaml" \
    --output_dir="./experiments/stage1_single1" \
    --resolution=512 \
    --train_batch_size=1 \
    --enable_xformers_memory_efficient_attention \
    --viz_freq=25 \
    --pretrained_path="./experiments/stage1_single/checkpoints/model_29001.pkl"
