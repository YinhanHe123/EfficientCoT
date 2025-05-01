#!/bin/bash

# Array of datasets
datasets=("gsm8k" "svamp" "multiarith" "commonsense_qa" "coin_flip")
models=("small" "mistral")

# Loop through each dataset
# for dataset in "${datasets[@]}"; do
#     for model in "${models[@]}"; do
#         python main.py --mode baseline --baseline ccot --config $model --device 3 --ccot_stage encode --dataset $dataset
#         python main.py --mode baseline --baseline ccot --config $model --device 3 --ccot_stage cotrain_encode_decode --dataset $dataset
#         python main.py --mode baseline --baseline ccot --config $model --device 3 --ccot_stage evaluate --eval_temp 0.7 --eval_max_contemp_tokens 1 --dataset $dataset
#     done
# done

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        CUDA_VISIBLE_DEVICES=2 python main.py --mode baseline --baseline pause --config $model --dataset $dataset --device 2
    done
done

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        python main.py --mode baseline --baseline codi --config $model --dataset $dataset --device 2
    done
done

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        python main.py --mode baseline --baseline icot_si --config $model --device 1 --dataset $dataset
    done
done