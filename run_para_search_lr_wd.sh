#!/bin/bash

# Grid Search Script for Efficient CoT project

# Define configs and datasets to test
CONFIGS=("mistral" "small")
DATASETS=("svamp" "gsm8k" "multiarith")

# Parameter grids - adjust these values based on your expected ranges
SENT_TRANS_LR_VALUES=(1e-5 3e-5 1e-4)
SENT_TRANS_WD_VALUES=(0.001 0.01 0.1)
# Fixed epochs for sentence transformer
SENT_TRANS_EPOCHS=15

CONTEMP_GEN_LR_VALUES=(1e-7 1e-6 1e-5)
CONTEMP_GEN_WD_VALUES=(1e-6 1e-5 1e-4)
CONTEMP_GEN_EPOCHS_VALUES=(2 3 4)

CONTEMP_GEN_LL_LR_VALUES=(0.0005 0.001 0.005)
CONTEMP_GEN_LL_WD_VALUES=(0.0005 0.001 0.005)
# Fixed epochs for linear layer
CONTEMP_GEN_LL_EPOCHS=10

# Evaluation parameters
EVAL_MAX_CONTEMP_TOKENS=(1 2 3 4 5)
EVAL_TEMPERATURES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Create logs directory for the grid search
GRID_SEARCH_LOGS="./grid_search_logs"
mkdir -p "$GRID_SEARCH_LOGS"

# Function to run a complete experiment with given parameters
run_experiment() {
    CONFIG=$1
    DATASET=$2
    SENT_TRANS_LR=$3
    SENT_TRANS_WD=$4
    CONTEMP_GEN_LR=$5
    CONTEMP_GEN_WD=$6
    CONTEMP_GEN_EPOCHS=$7
    CONTEMP_GEN_LL_LR=$8
    CONTEMP_GEN_LL_WD=$9

    # Create a unique experiment ID
    EXP_ID="config_${CONFIG}_dataset_${DATASET}_sentlr_${SENT_TRANS_LR}_sentwd_${SENT_TRANS_WD}_contemlr_${CONTEMP_GEN_LR}_contemwd_${CONTEMP_GEN_WD}_contemepochs_${CONTEMP_GEN_EPOCHS}_llr_${CONTEMP_GEN_LL_LR}_lwd_${CONTEMP_GEN_LL_WD}"
    LOG_FILE="${GRID_SEARCH_LOGS}/${EXP_ID}.log"

    echo "Starting experiment: $EXP_ID"
    echo "Experiment details:" > "$LOG_FILE"
    echo "  Config: $CONFIG" >> "$LOG_FILE"
    echo "  Dataset: $DATASET" >> "$LOG_FILE"
    echo "  Sentence Transformer LR: $SENT_TRANS_LR" >> "$LOG_FILE"
    echo "  Sentence Transformer WD: $SENT_TRANS_WD" >> "$LOG_FILE"
    echo "  Sentence Transformer Epochs: $SENT_TRANS_EPOCHS" >> "$LOG_FILE"
    echo "  Contemplation Generator LR: $CONTEMP_GEN_LR" >> "$LOG_FILE"
    echo "  Contemplation Generator WD: $CONTEMP_GEN_WD" >> "$LOG_FILE"
    echo "  Contemplation Generator Epochs: $CONTEMP_GEN_EPOCHS" >> "$LOG_FILE"
    echo "  Contemplation Generator Linear Layer LR: $CONTEMP_GEN_LL_LR" >> "$LOG_FILE"
    echo "  Contemplation Generator Linear Layer WD: $CONTEMP_GEN_LL_WD" >> "$LOG_FILE"
    echo "  Contemplation Generator Linear Layer Epochs: $CONTEMP_GEN_LL_EPOCHS" >> "$LOG_FILE"
    echo "=======================================" >> "$LOG_FILE"

    # Step 1: Train sentence transformer
    echo "Training sentence transformer..." | tee -a "$LOG_FILE"
    python main.py --mode train_sentence_transformer \
                  --config "$CONFIG" \
                  --dataset "$DATASET" \
                  --sent_trans_lr "$SENT_TRANS_LR" \
                  --sent_trans_weight_decay "$SENT_TRANS_WD" \
                  --sent_trans_epochs "$SENT_TRANS_EPOCHS" \
                  --variation "vanilla" \
                  --baseline "effi_cot" 2>&1 | tee -a "$LOG_FILE"

    # Check if the sentence transformer training was successful
    if [ $? -ne 0 ]; then
        echo "Error: Sentence transformer training failed. Skipping rest of experiment." | tee -a "$LOG_FILE"
        return 1
    fi

    # Step 2: Train contemplation generator
    echo "Training contemplation generator..." | tee -a "$LOG_FILE"
    python main.py --mode train_contemp_generator \
                  --config "$CONFIG" \
                  --dataset "$DATASET" \
                  --contemp_gen_lr "$CONTEMP_GEN_LR" \
                  --contemp_gen_weight_decay "$CONTEMP_GEN_WD" \
                  --contemp_gen_epochs "$CONTEMP_GEN_EPOCHS" \
                  --contemp_gen_lin_layer_lr "$CONTEMP_GEN_LL_LR" \
                  --contemp_gen_lin_layer_weight_decay "$CONTEMP_GEN_LL_WD" \
                  --contemp_gen_lin_layer_epochs "$CONTEMP_GEN_LL_EPOCHS" \
                  --variation "vanilla" \
                  --baseline "effi_cot" 2>&1 | tee -a "$LOG_FILE"

    # Check if the contemplation generator training was successful
    if [ $? -ne 0 ]; then
        echo "Error: Contemplation generator training failed. Skipping evaluation." | tee -a "$LOG_FILE"
        return 1
    fi

    # Step 3: Evaluate with different combinations of eval_max_contemp_tokens and temperatures
    echo "Running evaluations with different parameters..." | tee -a "$LOG_FILE"

    # Create evaluation metrics file
    EVAL_METRICS_FILE="${GRID_SEARCH_LOGS}/${EXP_ID}_eval_metrics.csv"
    echo "config,dataset,sent_trans_lr,sent_trans_wd,contemp_gen_lr,contemp_gen_wd,contemp_gen_epochs,contemp_gen_ll_lr,contemp_gen_ll_wd,eval_max_contemp_tokens,eval_temp,numerical_accuracy,close_match_rate,mean_relative_error,median_relative_error" > "$EVAL_METRICS_FILE"

    # Track best evaluation metrics
    BEST_ACCURACY=0
    BEST_TOKENS=0
    BEST_TEMP=0

    for EVAL_TOKENS in "${EVAL_MAX_CONTEMP_TOKENS[@]}"; do
        for EVAL_TEMP in "${EVAL_TEMPERATURES[@]}"; do
            echo "Evaluating with eval_max_contemp_tokens=$EVAL_TOKENS, eval_temp=$EVAL_TEMP" | tee -a "$LOG_FILE"

            EVAL_LOG="${GRID_SEARCH_LOGS}/${EXP_ID}_eval_tokens${EVAL_TOKENS}_temp${EVAL_TEMP}.log"

            python main.py --mode evaluate \
                          --config "$CONFIG" \
                          --dataset "$DATASET" \
                          --variation "vanilla" \
                          --baseline "effi_cot" \
                          --eval_max_contemp_tokens "$EVAL_TOKENS" \
                          --eval_temp "$EVAL_TEMP" 2>&1 | tee "$EVAL_LOG"

            # Extract metrics from the evaluation log
            NUM_ACCURACY=$(grep -o "numerical_accuracy\":[^,]*" "$EVAL_LOG" | tail -1 | cut -d ":" -f2)
            CLOSE_MATCH=$(grep -o "close_match_rate\":[^,]*" "$EVAL_LOG" | tail -1 | cut -d ":" -f2)
            MEAN_ERROR=$(grep -o "mean_relative_error\":[^,]*" "$EVAL_LOG" | tail -1 | cut -d ":" -f2)
            MEDIAN_ERROR=$(grep -o "median_relative_error\":[^,]*" "$EVAL_LOG" | tail -1 | cut -d ":" -f2)

            # Append to evaluation metrics file
            echo "$CONFIG,$DATASET,$SENT_TRANS_LR,$SENT_TRANS_WD,$CONTEMP_GEN_LR,$CONTEMP_GEN_WD,$CONTEMP_GEN_EPOCHS,$CONTEMP_GEN_LL_LR,$CONTEMP_GEN_LL_WD,$EVAL_TOKENS,$EVAL_TEMP,$NUM_ACCURACY,$CLOSE_MATCH,$MEAN_ERROR,$MEDIAN_ERROR" >> "$EVAL_METRICS_FILE"

            # Check if this is the best accuracy so far
            if (( $(echo "$NUM_ACCURACY > $BEST_ACCURACY" | bc -l) )); then
                BEST_ACCURACY=$NUM_ACCURACY
                BEST_TOKENS=$EVAL_TOKENS
                BEST_TEMP=$EVAL_TEMP
            fi

            # Append summary of this evaluation to main log
            echo "  Tokens: $EVAL_TOKENS, Temp: $EVAL_TEMP - Accuracy: $NUM_ACCURACY, Close Match: $CLOSE_MATCH" | tee -a "$LOG_FILE"
        done
    done

    # Add the best evaluation combination to the main experiment log
    echo "Best evaluation: Tokens=$BEST_TOKENS, Temp=$BEST_TEMP, Accuracy=$BEST_ACCURACY" | tee -a "$LOG_FILE"

    # Extract and record the best performance metrics for the main results summary
    METRICS_FILE="${GRID_SEARCH_LOGS}/results_summary.csv"

    # Create header if file doesn't exist
    if [ ! -f "$METRICS_FILE" ]; then
        echo "config,dataset,sent_trans_lr,sent_trans_wd,contemp_gen_lr,contemp_gen_wd,contemp_gen_epochs,contemp_gen_ll_lr,contemp_gen_ll_wd,best_tokens,best_temp,numerical_accuracy,close_match_rate,mean_relative_error,median_relative_error" > "$METRICS_FILE"
    fi

    # Get the row with the best accuracy from the evaluation metrics file
    BEST_EVAL=$(sort -t, -k12 -nr "$EVAL_METRICS_FILE" | head -1)

    # Append to summary file
    echo "$BEST_EVAL" >> "$METRICS_FILE"

    return 0
}

# Main grid search loop
for CONFIG in "${CONFIGS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "=============================================="
        echo "Starting grid search for config=$CONFIG, dataset=$DATASET"
        echo "=============================================="

        # Run a reduced grid search based on previous findings
        # We'll focus on sentence transformer parameters first since they impact downstream
        for SENT_TRANS_LR in "${SENT_TRANS_LR_VALUES[@]}"; do
            for SENT_TRANS_WD in "${SENT_TRANS_WD_VALUES[@]}"; do
                # Use middle values for other parameters in initial search
                CONTEMP_GEN_LR=${CONTEMP_GEN_LR_VALUES[1]}
                CONTEMP_GEN_WD=${CONTEMP_GEN_WD_VALUES[1]}
                CONTEMP_GEN_EPOCHS=${CONTEMP_GEN_EPOCHS_VALUES[1]}
                CONTEMP_GEN_LL_LR=${CONTEMP_GEN_LL_LR_VALUES[1]}
                CONTEMP_GEN_LL_WD=${CONTEMP_GEN_LL_WD_VALUES[1]}

                run_experiment "$CONFIG" "$DATASET" "$SENT_TRANS_LR" "$SENT_TRANS_WD" \
                               "$CONTEMP_GEN_LR" "$CONTEMP_GEN_WD" "$CONTEMP_GEN_EPOCHS" \
                               "$CONTEMP_GEN_LL_LR" "$CONTEMP_GEN_LL_WD"
            done
        done

        # Find best sentence transformer parameters from initial search
        echo "Finding best sentence transformer parameters..."
        BEST_SENT_PARAMS=$(grep "$CONFIG,$DATASET" "${GRID_SEARCH_LOGS}/results_summary.csv" | sort -t, -k12 -nr | head -1)
        BEST_SENT_LR=$(echo "$BEST_SENT_PARAMS" | cut -d, -f3)
        BEST_SENT_WD=$(echo "$BEST_SENT_PARAMS" | cut -d, -f4)

        echo "Best sentence transformer parameters: LR=$BEST_SENT_LR, WD=$BEST_SENT_WD"

        # Now optimize contemplation generator parameters
        for CONTEMP_GEN_LR in "${CONTEMP_GEN_LR_VALUES[@]}"; do
            for CONTEMP_GEN_WD in "${CONTEMP_GEN_WD_VALUES[@]}"; do
                # Use middle value for epochs
                CONTEMP_GEN_EPOCHS=${CONTEMP_GEN_EPOCHS_VALUES[1]}
                # Use middle values for linear layer params
                CONTEMP_GEN_LL_LR=${CONTEMP_GEN_LL_LR_VALUES[1]}
                CONTEMP_GEN_LL_WD=${CONTEMP_GEN_LL_WD_VALUES[1]}

                run_experiment "$CONFIG" "$DATASET" "$BEST_SENT_LR" "$BEST_SENT_WD" \
                               "$CONTEMP_GEN_LR" "$CONTEMP_GEN_WD" "$CONTEMP_GEN_EPOCHS" \
                               "$CONTEMP_GEN_LL_LR" "$CONTEMP_GEN_LL_WD"
            done
        done

        # Find best contemp generator params
        echo "Finding best contemplation generator parameters..."
        BEST_CONTEMP_PARAMS=$(grep "$CONFIG,$DATASET" "${GRID_SEARCH_LOGS}/results_summary.csv" | sort -t, -k12 -nr | head -1)
        BEST_CONTEMP_LR=$(echo "$BEST_CONTEMP_PARAMS" | cut -d, -f5)
        BEST_CONTEMP_WD=$(echo "$BEST_CONTEMP_PARAMS" | cut -d, -f6)

        # Explore contemp generator epochs
        for CONTEMP_GEN_EPOCHS in "${CONTEMP_GEN_EPOCHS_VALUES[@]}"; do
            # Use middle values for linear layer params
            CONTEMP_GEN_LL_LR=${CONTEMP_GEN_LL_LR_VALUES[1]}
            CONTEMP_GEN_LL_WD=${CONTEMP_GEN_LL_WD_VALUES[1]}

            run_experiment "$CONFIG" "$DATASET" "$BEST_SENT_LR" "$BEST_SENT_WD" \
                           "$BEST_CONTEMP_LR" "$BEST_CONTEMP_WD" "$CONTEMP_GEN_EPOCHS" \
                           "$CONTEMP_GEN_LL_LR" "$CONTEMP_GEN_LL_WD"
        done

        # Find best contemp generator configuration
        echo "Finding best overall contemplation generator configuration..."
        BEST_CONTEMP_CONFIG=$(grep "$CONFIG,$DATASET" "${GRID_SEARCH_LOGS}/results_summary.csv" | sort -t, -k12 -nr | head -1)
        BEST_CONTEMP_LR=$(echo "$BEST_CONTEMP_CONFIG" | cut -d, -f5)
        BEST_CONTEMP_WD=$(echo "$BEST_CONTEMP_CONFIG" | cut -d, -f6)
        BEST_CONTEMP_EPOCHS=$(echo "$BEST_CONTEMP_CONFIG" | cut -d, -f7)

        # Now optimize the linear layer params
        for CONTEMP_GEN_LL_LR in "${CONTEMP_GEN_LL_LR_VALUES[@]}"; do
            for CONTEMP_GEN_LL_WD in "${CONTEMP_GEN_LL_WD_VALUES[@]}"; do
                run_experiment "$CONFIG" "$DATASET" "$BEST_SENT_LR" "$BEST_SENT_WD" \
                               "$BEST_CONTEMP_LR" "$BEST_CONTEMP_WD" "$BEST_CONTEMP_EPOCHS" \
                               "$CONTEMP_GEN_LL_LR" "$CONTEMP_GEN_LL_WD"
            done
        done

        # Determine final best configuration
        echo "Determining final best configuration for $CONFIG - $DATASET..."
        BEST_CONFIG=$(grep "$CONFIG,$DATASET" "${GRID_SEARCH_LOGS}/results_summary.csv" | sort -t, -k12 -nr | head -1)

        echo "================ BEST CONFIGURATION ================="
        echo "Config: $CONFIG"
        echo "Dataset: $DATASET"
        echo "Sentence Transformer LR: $(echo "$BEST_CONFIG" | cut -d, -f3)"
        echo "Sentence Transformer WD: $(echo "$BEST_CONFIG" | cut -d, -f4)"
        echo "Contemplation Generator LR: $(echo "$BEST_CONFIG" | cut -d, -f5)"
        echo "Contemplation Generator WD: $(echo "$BEST_CONFIG" | cut -d, -f6)"
        echo "Contemplation Generator Epochs: $(echo "$BEST_CONFIG" | cut -d, -f7)"
        echo "Linear Layer LR: $(echo "$BEST_CONFIG" | cut -d, -f8)"
        echo "Linear Layer WD: $(echo "$BEST_CONFIG" | cut -d, -f9)"
        echo "Best Eval Tokens: $(echo "$BEST_CONFIG" | cut -d, -f10)"
        echo "Best Eval Temperature: $(echo "$BEST_CONFIG" | cut -d, -f11)"
        echo "Numerical Accuracy: $(echo "$BEST_CONFIG" | cut -d, -f12)"
        echo "=============================================="

        # Save best configuration
        echo "$BEST_CONFIG" > "${GRID_SEARCH_LOGS}/best_config_${CONFIG}_${DATASET}.csv"
    done
done

echo "Grid search completed. Results are in ${GRID_SEARCH_LOGS}/results_summary.csv"
echo "Best configurations are saved in ${GRID_SEARCH_LOGS}/best_config_*.csv"