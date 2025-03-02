#!/bin/bash

# Get the number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Found ${NUM_GPUS} GPUs available for training"

# Default values
MODE="train_sentence_transformer"
CONFIG="large"
BASELINE="effi_cot"
VARIATION="vanilla"
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --baseline)
      BASELINE="$2"
      shift 2
      ;;
    --variation)
      VARIATION="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Starting distributed training with ${NUM_GPUS} GPUs"
echo "Mode: ${MODE}"
echo "Config: ${CONFIG}"
echo "Baseline: ${BASELINE}"
echo "Variation: ${VARIATION}"
echo "Seed: ${SEED}"

# Set memory management options for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Launch distributed training using torchrun with memory optimizations
torchrun --nproc_per_node=${NUM_GPUS} \
         --master_port=12355 \
         launch_distributed.py \
         --mode=${MODE} \
         --config=${CONFIG} \
         --baseline=${BASELINE} \
         --variation=${VARIATION} \
         --seed=${SEED}