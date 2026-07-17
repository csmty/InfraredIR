#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

SD_TURBO_PATH="/path/to/sd-turbo"
PRETRAINED_PATH="/path/to/stage1_or_base_checkpoint.pkl"
OUTPUT_DIR="./experiments/stage1_single"

accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 29300 src/train.py \
    --sd_path="${SD_TURBO_PATH}" \
    --base_config="configs/train_STAGE1_single.yaml" \
    --output_dir="${OUTPUT_DIR}" \
    --resolution=512 \
    --train_batch_size=1 \
    --enable_xformers_memory_efficient_attention \
    --viz_freq=25 \
    --pretrained_path="${PRETRAINED_PATH}"
