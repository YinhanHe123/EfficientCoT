#!/bin/bash
export TMPDIR=/dev/shm
# Define arrays for methods and datasets
export CUDA_VISIBLE_DEVICES=1,2
methods=("icot_kd" "icot_si" "codi" "softcot" "coconut" "ccot" "pause")
datasets=("gsm8k" "svamp" "multiarith" "commonsense_qa" "coin_flip")
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

    if [[ "$method" == "ccot" ]]; then
        # For ccot, run all stages sequentially
        for stage in "${ccot_stages[@]}"; do
            run_on_gpu $gpu_id "python main.py --dataset $dataset --mode baseline --baseline ccot --ccot_stage $stage"
        done
    else
        # For other methods
        run_on_gpu $gpu_id "python main.py --dataset $dataset --mode baseline --baseline $method"
    fi
}

# Queue to manage tasks
declare -a gpu0_queue
declare -a gpu1_queue

# Distribute tasks between GPUs
task_count=0
for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        if (( task_count % 2 == 0 )); then
            gpu0_queue+=("run_method 0 $method $dataset")
        else
            gpu1_queue+=("run_method 1 $method $dataset")
        fi
        ((task_count++))
    done
done

# Run tasks on GPU 0
for cmd in "${gpu0_queue[@]}"; do
    eval "$cmd"
done &

# Run tasks on GPU 1
for cmd in "${gpu1_queue[@]}"; do
    eval "$cmd"
done &

# Wait for both GPUs to complete
wait
echo "All tasks completed!"