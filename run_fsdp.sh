#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_fsdp_${TIMESTAMP}.log"

# Set distributed training environment variables
export WORLD_SIZE=1
# export RANK=0
# export LOCAL_RANK=0
export NCCL_P2P_DISABLE=1
export WANDB_API_KEY="e5cbd387d8e0c181e93c7e4ec56e965c5115e94c"
export WANDB_CONFIG_DIR="/home/user1/.config/wandb_arash"
export WANDB_CACHE_DIR="/home/user1/.cache/wandb_arash"

# GPU Configuration
# Specify which GPUs to use (comma-separated list, e.g., "0,1,2,3" for first 4 GPUs)
# Leave empty to use all available GPUs
export CUDA_VISIBLE_DEVICES="2,3,4,5"  # Modify this line to select specific GPUs

# Number of GPUs to use (should match the number of GPUs in CUDA_VISIBLE_DEVICES)
NUM_GPUS=4  # Modify this to match the number of GPUs you want to use

# Run the training command and redirect all output to the log file
echo "Starting FSDP training at $(date)" | tee -a "$LOG_FILE"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "Command: torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS scripts/pretrain.py --model.type \"PRISM_QWEN25_EXTRA_DINOSIGLIP_224PX_0_5B\" --wandb_project \"testing_vlm_training\" --wandb_entity \"arash-akbari-stu-northeastern-university\"" | tee -a "$LOG_FILE"

# Run the command and redirect both stdout and stderr to the log file
torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS scripts/pretrain.py \
  --model.type "PRISM_QWEN25_EXTRA_DINOSIGLIP_224PX_0_5B" \
  --wandb_project "testing_vlm_training" \
  --wandb_entity "arash-akbari-stu-northeastern-university" \
  --model.enable_mixed_precision_training True   2>&1 | tee -a "$LOG_FILE"

# Check the exit status
if [ $? -eq 0 ]; then
    echo "FSDP training completed successfully at $(date)" | tee -a "$LOG_FILE"
else
    echo "FSDP training failed at $(date)" | tee -a "$LOG_FILE"
fi 