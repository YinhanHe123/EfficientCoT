#!/bin/bash
MODES=("train_sentence_transformer" "train_contemp_generator" "evaluate")
# VARIATIONS=("vanilla" "no_sentence_transformer" "no_l_reason")
VARIATIONS=("vanilla")
config="mistral"
dataset="multiarith"

for mode in "${MODES[@]}"; do
    for variation in "${VARIATIONS[@]}"; do
        echo "Running: python main.py --mode $mode --variation $variation --config $config --device 0 --dataset $dataset"
        python main.py --mode $mode --variation $variation --config $config --device 0 --dataset $dataset

        # Force CUDA cleanup between runs
        python -c "import torch; torch.cuda.empty_cache()"

        # Wait a few seconds to ensure memory is released
        sleep 5

        echo "----------------------------------------"
    done
done



# Define the parameter ranges
MAX_CONTEMP_TOKENS=(1 2 3 4 5)
EVAL_TEMPS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Track total runs and current progress
TOTAL_RUNS=$((${#MAX_CONTEMP_TOKENS[@]} * ${#EVAL_TEMPS[@]}))
CURRENT_RUN=0

echo "Starting evaluation runs with ${TOTAL_RUNS} combinations..."

# Loop through all combinations
for tokens in "${MAX_CONTEMP_TOKENS[@]}"; do
  for temp in "${EVAL_TEMPS[@]}"; do
    # Increment counter
    CURRENT_RUN=$((CURRENT_RUN + 1))

    # Display progress
    echo "[$CURRENT_RUN/$TOTAL_RUNS] Running with max_contemp_tokens=$tokens, eval_temp=$temp for dataset $dataset, config $config"

    # Run the Python command
    python main.py --mode evaluate --device 0 --variation vanilla --max_contemp_tokens $tokens --eval_temp $temp --dataset $dataset --config $config

    # Optional: Add a small delay between runs to avoid potential issues
    sleep 1
  done
done

echo "All combinations completed for dataset $dataset and config $config."