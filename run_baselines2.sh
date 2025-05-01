#!/bin/bash

# Array of datasets
datasets=("gsm8k" "svamp" "multiarith" "commonsense_qa" "coin_flip")
models=("small" "mistral")

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        python main.py --mode baseline --baseline softcot --config $model --device 1 --dataset $dataset
    done
done

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        python main.py --mode baseline --baseline coconut --coconut_stage train --config $model --device 1 --dataset $dataset
        python main.py --mode baseline --baseline coconut --coconut_stage evaluate --config $model --device 1 --dataset $dataset
    done
done

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        python main.py --mode baseline --baseline ccot --ccot_stage encode --config $model --device 1 --dataset $dataset
        python main.py --mode baseline --baseline ccot --ccot_stage decode --config $model --device 1 --dataset $dataset
        python main.py --mode baseline --baseline ccot --ccot_stage evaluate --config $model --device 1 --dataset $dataset
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