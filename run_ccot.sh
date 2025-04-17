#!/bin/bash

# Array of datasets
datasets=("svamp" "gsm8k" "multiarith")

# Array of temperatures
# temperatures=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
temperatures=(0.7)

# Array of max contemp tokens
# max_contemp_tokens=(1 2 3 4 5)
max_contemp_tokens=(1)

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    echo "============================================"
    echo "Starting process for dataset: $dataset"
    echo "============================================"

    # Run the training command

    echo "Running training for $dataset..."
    python main.py --mode baseline --baseline ccot --config mistral --device 0 --ccot_stage encode --dataset $dataset
    python main.py --mode baseline --baseline ccot --config mistral --device 0 --ccot_stage cotrain_encode_decode --dataset $dataset

    # Check if the training command was successful
    if [ $? -ne 0 ]; then
        echo "Training failed for dataset $dataset! Skipping evaluation."
        continue
    fi

    echo "Training completed for $dataset. Starting evaluations..."

    # Loop through each temperature
    for temp in "${temperatures[@]}"; do
        # Loop through each max contemp token
        for token in "${max_contemp_tokens[@]}"; do
            echo "Evaluating with temperature $temp and max_contemp_tokens $token"

            # Run the evaluation command
            python main.py --mode baseline --baseline ccot --config mistral --device 0 --ccot_stage evaluate --eval_temp $temp --eval_max_contemp_tokens $token --dataset $dataset

            # Check if the evaluation was successful
            if [ $? -ne 0 ]; then
                echo "Evaluation failed for dataset $dataset with temp=$temp and tokens=$token!"
            else
                echo "Evaluation completed for dataset $dataset with temp=$temp and tokens=$token"
            fi

            echo "----------------------------------------"
        done
    done

    echo "All evaluations completed for dataset: $dataset"
    echo "============================================"
    echo ""
done

echo "All training and evaluations completed!"