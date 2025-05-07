#!/bin/bash
export TMPDIR=/dev/shm
# Define arrays for methods and datasets
export CUDA_VISIBLE_DEVICES=1,2
datasets=("commonsense_qa" "coin_flip" "gsm8k" "svamp" "multiarith")
models=("small" "mistral")

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
    local dataset=$2
    local model=$3

    run_on_gpu $gpu_id "python main.py --dataset $dataset --mode effi_cot --config $model -stle 1 -stllme 1 -cgle 1 -cgllme 1"
}

# Queue to manage tasks
declare -a gpu0_queue

# Distribute tasks between GPUs
task_count=0
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        gpu0_queue+=("run_method 1 $dataset $model")
        ((task_count++))
    done
done

# Run tasks on GPU 0
for cmd in "${gpu0_queue[@]}"; do
    eval "$cmd"
done &


# Wait for both GPUs to complete
wait
echo "All tasks completed!"