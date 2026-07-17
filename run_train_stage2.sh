#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 29300 src/train.py \
    --sd_path="/data/matengyu/code/@submit/Infrared/CVPR26/S3Diff/weight/sd-turbo/models--stabilityai--sd-turbo/snapshots/sd-turbo" \
    --base_config="configs/train_STAGE2.yaml" \
    --pretrained_path="./experiments/stage1_single1/checkpoints/model_1001.pkl" \
    --output_dir="./experiments/stage2_finetune" \
    --yolo_weight="/data/matengyu/code/@submit/Infrared/CVPR26/ReleaseCode/mty/InfraredIR/weight/yolo/best.pt" \
    --sctransnet_weight="/data/matengyu/code/@submit/Infrared/CVPR26/ReleaseCode/mty/InfraredIR/weight/SCTransNet/SCTransNet_NUAA_NUDT_IRSTD1K.pth.tar" \
    --segformer_config="./nets/segformer/configs/fmb.yaml" \
    --segformer_weight="/data/matengyu/code/@submit/Infrared/CVPR26/ReleaseCode/mty/InfraredIR/weight/segformer/best_model.pth" \
    --resolution=512 \
    --train_batch_size=1 \
    --enable_xformers_memory_efficient_attention \
    --viz_freq=25
