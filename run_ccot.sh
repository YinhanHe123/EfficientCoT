#!/bin/bash

# Define the stages to run sequentially
export CUDA_VISIBLE_DEVICES=0,2,3
# rm -r /data/nee7ne/effi_cot/saved_models/ccot
stages=("encode" "prepare_decode_data" "decode" "evaluate")

# Define the base command
base_cmd="python main.py --mode baseline --baseline ccot --config small --device 0"

# Loop through each stage and run the command
for stage in "${stages[@]}"; do
    echo "Running stage: $stage"
    cmd="$base_cmd --ccot_stage $stage"
    echo "Executing: $cmd"

    # Execute the command
    eval $cmd

    # Check if the command executed successfully
    if [ $? -eq 0 ]; then
        echo "Stage '$stage' completed successfully."
    else
        echo "Stage '$stage' failed with exit code $?. Stopping execution."
        exit 1
    fi

    echo "----------------------------------------"
done

echo "All stages completed successfully!"