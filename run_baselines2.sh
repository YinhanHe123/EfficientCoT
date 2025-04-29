#!/bin/bash

# Array of datasets
datasets=("svamp" "gsm8k" "multiarith")
models=("small" "mistral")

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        python main.py --mode baseline --baseline icot_si --config $model --device 1 --dataset $dataset
    done
done

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        python main.py --mode baseline --baseline codi --config $model --device 3 --dataset $dataset
    done
done

# for dataset in "${datasets[@]}"; do
#     for model in "${models[@]}"; do
#         python main.py --mode baseline --baseline pause --config $model --device 1 --dataset $dataset
#     done
# done

# for dataset in "${datasets[@]}"; do
#     for model in "${models[@]}"; do
#         python main.py --mode baseline --baseline softcot --config $model --device 1 --dataset $dataset
#     done
# done