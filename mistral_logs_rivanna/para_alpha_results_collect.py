import os
import pandas as pd
import glob
import re

# Base directory
base_dir = "/home/nee7ne/EfficientCoT/mistral_logs_rivanna"

# Updated pattern to catch all dataset names properly
dataset_pattern = re.compile(r"([a-zA-Z0-9_]+)_alpha_\d+\.\d+")

# Find all alpha directories
alpha_dirs = glob.glob(os.path.join(base_dir, "*_alpha_*"))

# Let's add some debug prints to see what's happening
print(f"Found {len(alpha_dirs)} alpha directories")
for dir_path in alpha_dirs:
    dir_name = os.path.basename(dir_path)
    print(f"Testing directory: {dir_name}")
    match = dataset_pattern.match(dir_name)
    print(f"Match result: {match}")

# Extract unique dataset names
dataset_names = set()
for dir_path in alpha_dirs:
    dir_name = os.path.basename(dir_path)
    match = dataset_pattern.match(dir_name)
    if match:
        dataset_names.add(match.group(1))
    else:
        print(f"WARNING: Could not extract dataset name from directory: {dir_name}")

print(f"Found datasets: {dataset_names}")

# Dictionary to store all dataframes by dataset
all_dfs = {dataset: [] for dataset in dataset_names}

# Process all files
for dir_path in alpha_dirs:
    dir_name = os.path.basename(dir_path)
    match = dataset_pattern.match(dir_name)

    if match:
        dataset = match.group(1)
        # Find the alpha value from the directory name
        alpha_value = dir_name.replace(f"{dataset}_alpha_", "")

        # Look for the best results CSV file
        csv_pattern = os.path.join(dir_path, f"best_results_{dir_name}.csv")
        csv_files = glob.glob(csv_pattern)

        if not csv_files:
            print(f"WARNING: No CSV files found matching pattern {csv_pattern}")

        for file in csv_files:
            try:
                # Read the CSV
                df = pd.read_csv(file)

                # Add columns to identify dataset and alpha value
                df["dataset"] = dataset
                df["alpha"] = alpha_value

                # Add to the appropriate dataset list
                all_dfs[dataset].append(df)

                print(f"Read {file}")
            except Exception as e:
                print(f"Error reading {file}: {e}")

# Combine all dataframes by dataset
combined_dfs = []
for dataset, dfs in all_dfs.items():
    if dfs:
        # Concatenate all dataframes for this dataset
        dataset_df = pd.concat(dfs, ignore_index=True)
        # Sort by alpha value
        dataset_df = dataset_df.sort_values("alpha")
        combined_dfs.append(dataset_df)
    else:
        print(f"WARNING: No data found for dataset {dataset}")

# Combine all datasets into a single dataframe
if combined_dfs:
    result_df = pd.concat(combined_dfs, ignore_index=True)

    # Reorder columns to have dataset and alpha as first columns
    cols = result_df.columns.tolist()
    cols = ["dataset", "alpha"] + [col for col in cols if col not in ["dataset", "alpha"]]
    result_df = result_df[cols]

    # Save the combined dataframe
    output_path = os.path.join(base_dir, "combined_alpha_results.csv")
    result_df.to_csv(output_path, index=False)
    print(f"Combined results saved to {output_path}")
else:
    print("No data was found to combine.")