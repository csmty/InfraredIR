#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH}"

SD_TURBO_PATH="/path/to/sd-turbo"
PRETRAINED_PATH="/path/to/InfraredIR.pkl"
OUTPUT_DIR="./output/enh"

accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 29300 src/inference.py \
    --base_config="configs/test.yaml" \
    --sd_path="${SD_TURBO_PATH}" \
    --pretrained_path="${PRETRAINED_PATH}" \
    --output_dir="${OUTPUT_DIR}"
