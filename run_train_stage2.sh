#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

SD_TURBO_PATH="/path/to/sd-turbo"
PRETRAINED_PATH="/path/to/stage1_checkpoint.pkl"
YOLO_WEIGHT="/path/to/yolo/best.pt"
SCTRANSNET_WEIGHT="/path/to/SCTransNet.pth.tar"
SEGFORMER_CONFIG="./nets/segformer/configs/fmb.yaml"
SEGFORMER_WEIGHT="/path/to/segformer/best_model.pth"
OUTPUT_DIR="./experiments/stage2_finetune"

accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 29300 src/train.py \
    --sd_path="${SD_TURBO_PATH}" \
    --base_config="configs/train_STAGE2.yaml" \
    --pretrained_path="${PRETRAINED_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --yolo_weight="${YOLO_WEIGHT}" \
    --sctransnet_weight="${SCTRANSNET_WEIGHT}" \
    --segformer_config="${SEGFORMER_CONFIG}" \
    --segformer_weight="${SEGFORMER_WEIGHT}" \
    --resolution=512 \
    --train_batch_size=1 \
    --enable_xformers_memory_efficient_attention \
    --viz_freq=25
