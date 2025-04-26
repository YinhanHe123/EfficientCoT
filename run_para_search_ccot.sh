#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,2
# Define the parameter ranges
MAX_CONTEMP_TOKENS=(1 2 3 4 5)
# MAX_CONTEMP_TOKENS=(1)
EVAL_TEMPS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# DATA=("svamp" "gsm8k" "multiarith")
DATA=("multiarith")

# Track total runs and current progress
TOTAL_RUNS=$((${#MAX_CONTEMP_TOKENS[@]} * ${#EVAL_TEMPS[@]}))
CURRENT_RUN=0
config="mistral"

echo "Starting evaluation runs with ${TOTAL_RUNS} combinations..."

# Loop through all combinations
for data in "${DATA[@]}"; do
  for tokens in "${MAX_CONTEMP_TOKENS[@]}"; do
    for temp in "${EVAL_TEMPS[@]}"; do
      # Increment counter
      CURRENT_RUN=$((CURRENT_RUN + 1))

      # Display progress
      echo "[$CURRENT_RUN/$TOTAL_RUNS] Running with max_contemp_tokens=$tokens, eval_temp=$temp, dataset=$data, config=$config"

      # Run the Python command
      python main.py --config $config --dataset $data --mode baseline --baseline ccot --ccot_stage evaluate --device 0 --eval_max_contemp_tokens $tokens --eval_temp $temp

      # Optional: Add a small delay between runs to avoid potential issues
      sleep 1
    done
  done
done
echo "All evaluation runs completed!"


