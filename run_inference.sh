accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 29300 src/inference.py \
    --base_config="configs/test.yaml" \
    --sd_path="/path/to/InfraredIR/sd-turbo" \
    --pretrained_path="/path/to/InfraredIR_best.pkl" \
    --output_dir="./output/enh"