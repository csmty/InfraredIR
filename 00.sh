python tools/compare_weights.py \
  ./experiments/stage1_single1/checkpoints/model_1001.pkl \
  ./experiments/stage2_finetune/checkpoints/model_501.pkl \
  --topk 50 \
  --show-missing \
  --show-shape-mismatch