#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpuA100x8
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --job-name=train0
#SBATCH --account=bdsz-delta-gpu
#SBATCH --mem=64G
#SBATCH --nodes=1

echo STARTING at $(date)
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

git rev-parse HEAD

source /u/yangc9/finetune/qlora/qlora_env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --multi_gpu --num_processes=8 qlora.py \
  --model_name_or_path codellama/CodeLlama-13b-Instruct-hf \
  --dataset=/u/yangc9/finetune/final_data_multi_new.jsonl \
  --no_gradient_checkpointing \
  |& tee cl37.log


echo END at $(date)
