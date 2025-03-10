#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,2,3'

# Define variables
NUM_GPUS=3  # Adjust based on your system
CONFIG="large"  # Use 'large' for the larger model configuration
VARIATION="vanilla"
MODE="train_sentence_transformer"
NUM_STAGES=2  # Number of pipeline stages

# Function to print section headers
print_header() {
    echo "======================================================"
    echo "  $1"
    echo "======================================================"
}

# Function to run a DeepSpeed command with the specified arguments
run_deepspeed() {
    local mode=$1
    local config=$2
    local variation=$3
    local num_stages=$4

    print_header "Running $mode with config=$config, variation=$variation"

    # Use DeepSpeed launcher to run the command
    deepspeed --launcher_args '--world_info "{\"localhost\": [0, 2, 3]}"' \
        deepspeed_main.py \
        --mode $mode \
        --config $config \
        --variation $variation \
        --num_stages $num_stages \
        --use_deepspeed \
        --seed 42

    # Force CUDA cleanup between runs
    python -c "import torch; torch.cuda.empty_cache()"

    # Wait a few seconds to ensure memory is released
    sleep 5
}

# Check if a specific mode was provided as an argument
if [ $# -gt 0 ]; then
    MODE=$1
fi

# Check if a specific config was provided as an argument
if [ $# -gt 1 ]; then
    CONFIG=$2
fi

# Check if a specific variation was provided as an argument
if [ $# -gt 2 ]; then
    VARIATION=$3
fi

# Run the specified mode or full pipeline
if [ "$MODE" == "all" ]; then
    # Run complete pipeline
    run_deepspeed "train_sentence_transformer" $CONFIG $VARIATION $NUM_STAGES
    run_deepspeed "train_contemp_generator" $CONFIG $VARIATION $NUM_STAGES
    run_deepspeed "evaluate" $CONFIG $VARIATION $NUM_STAGES
else
    # Run just the specified mode
    run_deepspeed $MODE $CONFIG $VARIATION $NUM_STAGES
fi

print_header "All jobs completed"