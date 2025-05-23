import os
import re
import json
import numpy as np
import csv
from collections import defaultdict

def process_log_file(log_path):
    # Dictionary to store results by temperature
    results_by_temp = defaultdict(lambda: {'num_acc': [], 'ave_time': []})

    try:
        with open(log_path, 'r') as f:
            log_content = f.read()

        # Find all evaluation result blocks using regex
        eval_results = re.findall(r"Evaluation results: (\{.*?\})", log_content, re.DOTALL)

        for result_str in eval_results:
            try:
                # Convert the string representation of dict to an actual dict
                result = json.loads(result_str.replace("'", '"'))

                # Extract temperature and metrics
                temp = result.get('eval_temp')
                num_acc = result.get('numerical_accuracy')
                ave_time = result.get('ave_sample_time')

                if temp is not None and num_acc is not None and ave_time is not None:
                    results_by_temp[temp]['num_acc'].append(num_acc)
                    results_by_temp[temp]['ave_time'].append(ave_time)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing result: {e}")
                continue

        # Calculate stats for each temperature
        stats = {}
        for temp, metrics in results_by_temp.items():
            num_acc_values = metrics['num_acc']
            ave_time_values = metrics['ave_time']

            # Only calculate if we have collected values
            if num_acc_values and ave_time_values:
                stats[temp] = {
                    'num_acc_mean': np.mean(num_acc_values),
                    'num_acc_std': np.std(num_acc_values),
                    'ave_time_mean': np.mean(ave_time_values),
                    'ave_time_std': np.std(ave_time_values)
                }

        # Find temperature with highest mean numerical accuracy
        if stats:
            best_temp = max(stats.keys(), key=lambda t: stats[t]['num_acc_mean'])
            return best_temp, stats[best_temp]
        else:
            print(f"No valid data found in {log_path}")
            return None, None

    except Exception as e:
        print(f"Error processing {log_path}: {e}")
        return None, None

def save_to_csv(log_path, best_temp, stats):
    # Create output CSV path based on log file path
    output_dir = os.path.dirname(log_path)
    output_filename = f"best_results_{os.path.basename(output_dir)}.csv"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['log_file', 'best_temp', 'num_acc_mean', 'num_acc_std', 'ave_time_mean', 'ave_time_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({
            'log_file': os.path.basename(log_path),
            'best_temp': best_temp,
            'num_acc_mean': stats['num_acc_mean'],
            'num_acc_std': stats['num_acc_std'],
            'ave_time_mean': stats['ave_time_mean'],
            'ave_time_std': stats['ave_time_std']
        })

    print(f"Results saved to {output_path}")

def main():
    orig_path = "/home/nee7ne/EfficientCoT/mistral_logs_rivanna"

    # Walk through directory structure
    for root, dirs, files in os.walk(orig_path):
        for file in files:
            if file == "output.log":
                log_path = os.path.join(root, file)
                print(f"Processing {log_path}")

                best_temp, stats = process_log_file(log_path)

                if best_temp is not None and stats is not None:
                    save_to_csv(log_path, best_temp, stats)

if __name__ == "__main__":
    main()