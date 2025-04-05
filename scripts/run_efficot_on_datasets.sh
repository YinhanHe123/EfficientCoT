#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,2,3
MODES=("train_sentence_transformer" "train_contemp_generator" "evaluate")
DATASETS=("multiarith")

for data in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "Running: python main.py --mode $mode --dataset $data --config small --device 0"
        python main.py --mode $mode --dataset $data --config small --device 0
        # Force CUDA cleanup between runs
        python -c "import torch; torch.cuda.empty_cache()"
        # Wait a few seconds to ensure memory is released
        # sleep 5
        echo "----------------------------------------"
    done
done
echo "All combinations completed."