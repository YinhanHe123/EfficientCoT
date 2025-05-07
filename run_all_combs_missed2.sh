#!/bin/bash
export TMPDIR=/dev/shm
# Define arrays for methods and datasets
export CUDA_VISIBLE_DEVICES=1
# methods=("icot_si" "codi" "softcot" "coconut" "ccot" "pause")
methods=("coconut" "coconut" "coconut" "coconut" "coconut" "coconut" "coconut" "coconut")
datasets=("gsm8k" "svamp" "multiarith" "commonsense_qa" "coin_flip" "multiarith" "commonsense_qa" "coin_flip")
configs=("small" "small" "small" "small" "small" "mistral" "mistral" "mistral")
datasets2=("gsm8k" "svamp" "multiarith" "commonsense_qa" "coin_flip")

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
    local method=$2
    local dataset=$3
    local config=$4
    run_on_gpu $gpu_id "python main.py --dataset $dataset --mode baseline --baseline $method --config $config"
}

# Queue to manage tasks
declare -a gpu1_queue

# Distribute tasks between GPUs
for ((i=0; i < ${#methods[@]}; i++ )); do
    gpu1_queue+=("run_method 2 ${methods[i]} ${datasets[i]} ${configs[i]}")
done

for dataset in "${datasets2[@]}"; do
    gpu1_queue+=("run_method 2 softcot $dataset")
done

# Run tasks on GPU 0
for cmd in "${gpu1_queue[@]}"; do
    eval "$cmd"
done &

# Wait for both GPUs to complete
wait
echo "All tasks completed!"