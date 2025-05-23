#!/bin/bash
export TMPDIR=/dev/shm
# Define arrays for methods and datasets
export CUDA_VISIBLE_DEVICES=1,2
models=("small" "mistral")
datasets=("gsm8k" "svamp" "multiarith" "commonsense_qa" "coin_flip")

# Function to run a command on a specific GPU
run_on_gpu() {
    local gpu_id=$1
    local cmd=$2
    echo "Running on GPU $gpu_id: $cmd"
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd
}

# Function to run a method on a dataset on a specific GPU
run_method() {
    local gpu_id=$1
    local model=$2
    local dataset=$3
    local method=$4

    run_on_gpu $gpu_id "python main.py --dataset $dataset --mode baseline --baseline $method --config $model --num_exps 1"
}

run_method_efficot() {
    local gpu_id=$1
    local model=$2
    local dataset=$3

    run_on_gpu $gpu_id "python main.py --dataset $dataset --mode efficot --config $model --num_exps 1"
}

# Queue to manage tasks
declare -a gpu_queue

# Distribute tasks between GPUs
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        gpu_queue+=("run_method 3 $model $dataset coconut")
    done
done

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        gpu_queue+=("run_method 3 $model $dataset codi")
    done
done

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        gpu_queue+=("run_method_efficot 3 $model $dataset")
    done
done

# Run tasks on GPU 0
for cmd in "${gpu_queue[@]}"; do
    eval "$cmd"
done &

# Wait for both GPUs to complete
wait
echo "All tasks completed!"