#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
# Define the parameter ranges
MAX_CONTEMP_TOKENS=(1 2 3 4 5)
EVAL_TEMPS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
DATA=("multiarith" "svamp")

# Track total runs and current progress
TOTAL_RUNS=$((${#MAX_CONTEMP_TOKENS[@]} * ${#EVAL_TEMPS[@]}))
CURRENT_RUN=0

echo "Starting evaluation runs with ${TOTAL_RUNS} combinations..."

# Loop through all combinations
for data in "${DATA[@]}"; do
  for tokens in "${MAX_CONTEMP_TOKENS[@]}"; do
    for temp in "${EVAL_TEMPS[@]}"; do
      # Increment counter
      CURRENT_RUN=$((CURRENT_RUN + 1))

      # Display progress
      echo "[$CURRENT_RUN/$TOTAL_RUNS] Running with max_contemp_tokens=$tokens, eval_temp=$temp, dataset=$data, config=small, config=small"

      # Run the Python command
      python main.py --config small --dataset $data --mode evaluate --device 0 --variation vanilla --max_contemp_tokens $tokens --eval_temp $temp

      # Optional: Add a small delay between runs to avoid potential issues
      sleep 1
    done
  done
done
echo "All evaluation runs completed!"


