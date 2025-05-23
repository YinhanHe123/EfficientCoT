import re
import numpy as np
import os
import pandas as pd

def extract_values_from_log(log_path):
    numerical_accuracy_values = []
    ave_sample_time_values = []

    # Regular expressions to extract the values
    accuracy_pattern = re.compile(r"'numerical_accuracy': ([\d.e+-]+)")
    time_pattern = re.compile(r"'ave_sample_time': ([\d.e+-]+)")

    try:
        with open(log_path, 'r') as file:
            content = file.read()

            # Find all matches
            accuracy_matches = accuracy_pattern.findall(content)
            time_matches = time_pattern.findall(content)

            # Convert string values to float
            numerical_accuracy_values = [float(val) for val in accuracy_matches]
            ave_sample_time_values = [float(val) for val in time_matches]

    except Exception as e:
        print(f"Error reading or processing the file {log_path}: {e}")
        return None, None

    return numerical_accuracy_values, ave_sample_time_values

def compute_statistics(values, label):
    if not values or len(values) < 15:
        print(f"Warning: Not enough {label} values in the log (found {len(values)}, need at least 15)")
        return [(np.nan, np.nan)] * 5  # Return empty stats if not enough data

    # Keep only the first 15 values
    values = values[:15]

    # Calculate statistics for each group (i, i+5, i+10)
    result_pairs = []
    for i in range(5):
        group = [values[i], values[i+5], values[i+10]]
        mean_val = np.mean(group)
        std_val = np.std(group)
        result_pairs.append((mean_val, std_val))

    return result_pairs

def process_dataset_folders(base_path, dataset_folders):
    results = []

    for dataset in dataset_folders:
        log_path = os.path.join(base_path, dataset, "output.log")
        print(f"Processing {log_path}...")

        if not os.path.exists(log_path):
            print(f"Warning: Log file not found at {log_path}")
            continue

        # Extract values
        numerical_accuracy_values, ave_sample_time_values = extract_values_from_log(log_path)

        if numerical_accuracy_values is None or ave_sample_time_values is None:
            print(f"Failed to extract values from {log_path}")
            continue

        print(f"Found {len(numerical_accuracy_values)} accuracy values and {len(ave_sample_time_values)} time values in {dataset}")

        # Calculate statistics
        accuracy_stats = compute_statistics(numerical_accuracy_values, "numerical_accuracy")
        time_stats = compute_statistics(ave_sample_time_values, "ave_sample_time")

        # Add to results
        for i, ((acc_mean, acc_std), (time_mean, time_std)) in enumerate(zip(accuracy_stats, time_stats)):
            results.append({
                'dataset': dataset,
                'group': i+1,
                'accuracy_mean': acc_mean,
                'accuracy_std': acc_std,
                'sample_time_mean': time_mean,
                'sample_time_std': time_std
            })

    return results

def save_to_csv(results, output_path):
    if not results:
        print("No results to save.")
        return False

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(results)

    # Save to CSV
    try:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return False

def main():
    base_path = "/home/nee7ne/EfficientCoT/logs/"
    dataset_folders = ["vanilla_gsm8k"]
    output_csv = "alpha_025_variation_vanilla_results.csv"

    # Process all dataset folders
    results = process_dataset_folders(base_path, dataset_folders)

    # Save results to CSV
    success = save_to_csv(results, output_csv)

    if success:
        # Display summary
        print("\nSummary of results by dataset:")
        df = pd.DataFrame(results)
        summary = df.groupby('dataset')[['accuracy_mean', 'sample_time_mean']].mean()
        print(summary)

        # Create a more readable summary table
        print("\nDetailed summary by dataset and group:")
        pivot_table = df.pivot_table(
            index=['dataset', 'group'],
            values=['accuracy_mean', 'accuracy_std', 'sample_time_mean', 'sample_time_std'],
            aggfunc='mean'
        )
        print(pivot_table)

if __name__ == "__main__":
    main()