#!/bin/bash
MODES=("train_sentence_transformer" "train_contemp_generator" "evaluate")
# VARIATIONS=("vanilla" "no_sentence_transformer" "no_l_reason")
VARIATIONS=("vanilla")

for mode in "${MODES[@]}"; do
    for variation in "${VARIATIONS[@]}"; do
        echo "Running: python main.py --mode $mode --variation $variation --config small"
        python main.py --mode $mode --variation $variation --config small --device 2

        # Force CUDA cleanup between runs
        python -c "import torch; torch.cuda.empty_cache()"

        # Wait a few seconds to ensure memory is released
        sleep 5

        echo "----------------------------------------"
    done
done
echo "All combinations completed."