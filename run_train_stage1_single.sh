export PYTHONPATH="/path/to/InfraredIR-main:$PYTHONPATH"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 29300 src/train.py \
    --sd_path="/path/to/sd-turbo" \
    --base_config="configs/train_STAGE1_single.yaml" \
    --output_dir="./experiments/stage1_single" \
    --resolution=512 \
    --train_batch_size=1 \
    --enable_xformers_memory_efficient_attention \
    --viz_freq=25