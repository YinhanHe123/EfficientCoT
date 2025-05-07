#!/bin/bash
export TMPDIR=/dev/shm
# Define arrays for methods and datasets
export CUDA_VISIBLE_DEVICES=1
# methods=("icot_si" "codi" "softcot" "coconut" "ccot" "pause")
methods=("softcot" "softcot" "softcot" "softcot" "softcot" "softcot" "softcot" "codi" "codi" "codi" "codi" "codi" "coconut" "coconut")
datasets=("gsm8k" "svamp" "multiarith" "commonsense_qa" "svamp" "gsm8k" "coin_flip" "gsm8k" "svamp" "multiarith" "commonsense_qa" "coin_flip" "gsm8k" "svamp")
configs=("small" "small" "small" "small" "mistral" "mistral" "mistral" "mistral" "mistral" "mistral" "mistral" "mistral" "mistral" "mistral")
ccot_stages=("encode" "prepare_decode_data" "cotrain_encode_decode" "evaluate")

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
    gpu1_queue+=("run_method 1 ${methods[i]} ${datasets[i]} ${configs[i]}")
done

# Run tasks on GPU 0
for cmd in "${gpu1_queue[@]}"; do
    eval "$cmd"
done &

# Wait for both GPUs to complete
wait
echo "All tasks completed!"