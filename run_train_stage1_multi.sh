export PYTHONPATH="/path/to/InfraredIR-main:$PYTHONPATH"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 29300 src/train.py \
    --sd_path="/data/matengyu/code/@submit/Infrared/CVPR26/S3Diff/weight/sd-turbo/models--stabilityai--sd-turbo/snapshots/sd-turbo" \
    --base_config="configs/train_STAGE1_multi.yaml" \
    --output_dir="./experiments/stage1_multi" \
    --yolo_weight="/data/matengyu/code/@submit/Infrared/CVPR26/ReleaseCode/mty/InfraredIR/weight/yolo/best.pt" \
    --sctransnet_weight="/data/matengyu/code/@submit/Infrared/CVPR26/ReleaseCode/mty/InfraredIR/weight/SCTransNet/SCTransNet_NUAA_NUDT_IRSTD1K.pth.tar" \
    --segformer_config="./nets/segformer/configs/fmb.yaml" \
    --segformer_weight="/data/matengyu/code/@submit/Infrared/CVPR26/ReleaseCode/mty/InfraredIR/weight/segformer/best_model.pth" \
    --resolution=512 \
    --train_batch_size=1 \
    --enable_xformers_memory_efficient_attention \
    --viz_freq=25 \
    --pretrained_path="./weight/InfraredIR.pkl"