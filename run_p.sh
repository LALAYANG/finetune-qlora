echo STARTING at $(date)
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

git rev-parse HEAD

source /u/yangc9/finetune/qlora/qlora_env/bin/activate

#--multi_gpu --num_processes=2 
  # --no_gradient_checkpointing \
    # --bf16 True \   --bf16 True \
  # --per_device_train_batch_size 1 \
  # --per_device_eval_batch_size 1 \
  #  --gradient_checkpointing True \
# accelerate launch

# export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# accelerate launch --multi_gpu --num_processes=2 -- \
#   qlora.py \
#   --model_name_or_path deepseek-ai/deepseek-coder-6.7b-base \
#   --dataset=/u/yangc9/finetune/final_data_multi_tasks.jsonl \
#   --no_gradient_checkpointing  \
#   --per_device_train_batch_size 1 \
#  --per_device_eval_batch_size 1 \
#   |& tee t1.log

# accelerate launch --multi_gpu --num_processes=2 -- \

# export OMP_NUM_THREADS=4

# torchrun --nproc_per_node=2 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --multi_gpu --num_processes=6 qlora.py \
  --model_name_or_path deepseek-ai/deepseek-coder-33b-instruct \
  --dataset=/u/yangc9/finetune/final_data_multi_new.jsonl \
  --no_gradient_checkpointing \
  |& tee t33.log
    # --no_gradient_checkpointing \



  # --gradient_checkpointing True \
  # --bf16 True \
  # --per_device_train_batch_size 1 \
  # --max_train_samples 2000 \


echo END at $(date)
