#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

# Set distributed training environment variables
# export WORLD_SIZE=1
# export RANK=0
# export LOCAL_RANK=0
export NCCL_P2P_DISABLE=1
export WANDB_API_KEY="e5cbd387d8e0c181e93c7e4ec56e965c5115e94c"
export WANDB_CONFIG_DIR="/home/user1/.config/wandb_arash"
export WANDB_CACHE_DIR="/home/user1/.cache/wandb_arash"




# Run the training command and redirect all output to the log file
echo "Starting training at $(date)" | tee -a "$LOG_FILE"
echo "Command: torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py --model.type \"prism-qwen25-extra-dinosiglip-224px+0_5b\" --wandb_project \"testing_vlm_training\" --wandb_entity \"arash-akbari-stu-northeastern-university\"" | tee -a "$LOG_FILE"


# Run the command and redirect both stdout and stderr to the log file
torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
  --model.type "prism-qwen25-extra-dinosiglip-224px+0_5b" \
  --wandb_project "testing_vlm_training" \
  --wandb_entity "arash-akbari-stu-northeastern-university" 2>&1 | tee -a "$LOG_FILE"

# Check the exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)" | tee -a "$LOG_FILE"
else
    echo "Training failed at $(date)" | tee -a "$LOG_FILE"
fi
