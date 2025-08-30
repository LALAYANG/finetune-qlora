#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00
#SBATCH --job-name=train0
#SBATCH --account=bdsz-delta-gpu
#SBATCH --mem=64G
#SBATCH --nodes=1

echo STARTING at $(date)
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

git rev-parse HEAD

source /u/yangc9/finetune/qlora/qlora_env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --multi_gpu --num_processes=4 qlora.py \
  --model_name_or_path deepseek-ai/deepseek-coder-6.7b-base \
  --dataset=/u/yangc9/finetune/final_data_multi_new.jsonl \
  --no_gradient_checkpointing \
  |& tee d0.log


echo END at $(date)
