import os
import pandas as pd
import glob

# Base directory
base_dir = "/home/nee7ne/EfficientCoT/mistral_logs_rivanna"

# Identify all the variation directories
variation_dirs = [
    "alpha_025_variation_no_l_reason",
    "alpha_025_variation_no_sentence_transformer",
    "alpha_025_variation_no_small_contemp_gen"
]

# Dataset types to look for
datasets = ["coinflip", "comsense_qa", "gsm8k", "multiarith", "svamp"]

# Dictionary to store all dataframes by dataset
all_dfs = {dataset: [] for dataset in datasets}

# Collect all CSV files
for variation in variation_dirs:
    for dataset in datasets:
        # Pattern for the csv file
        pattern = os.path.join(base_dir, variation, f"data_{dataset}", "best_results_data_*.csv")
        csv_files = glob.glob(pattern)

        for file in csv_files:
            try:
                # Read the CSV
                df = pd.read_csv(file)

                # Add a column to identify which variation this data is from
                df["variation"] = variation

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
        # Add a column to identify the dataset
        dataset_df["dataset"] = dataset
        combined_dfs.append(dataset_df)

# Combine all datasets into a single dataframe
if combined_dfs:
    result_df = pd.concat(combined_dfs, ignore_index=True)

    # Reorder columns to have dataset and variation as first columns
    cols = result_df.columns.tolist()
    cols = ["dataset", "variation"] + [col for col in cols if col not in ["dataset", "variation"]]
    result_df = result_df[cols]

    # Save the combined dataframe
    output_path = os.path.join(base_dir, "ablation_combined_results.csv")
    result_df.to_csv(output_path, index=False)
    print(f"Combined results saved to {output_path}")
else:
    print("No data was found to combine.")