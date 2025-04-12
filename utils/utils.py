import os
import random
import numpy as np
import torch
import json
from pathlib import Path

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_json(data, filepath):
    """Save data as a JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath):
    """Load data from a JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_directory(file_path):
    """Create directory if it doesn't exist"""
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path) if not os.path.exists(dir_path) else None
    return

def format_time(seconds):
    """Format time in seconds to hours:minutes:seconds"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_lr(optimizer):
    """Get the current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_experiment_config(model_config, experiment_config, output_dir, experiment_name):
    """
    Save experiment configurations for reproducibility

    Args:
        model_config: Model configuration object
        experiment_config: Experiment configuration object
        output_dir: Directory to save the configuration
        experiment_name: Name of the experiment
    """
    create_directory(output_dir)

    # Combine configurations
    combined_config = {
        "experiment_name": experiment_name,
        "model_config": {k: v for k, v in model_config.__dict__.items()
                         if not k.startswith('_')},
        "experiment_config": {k: v for k, v in experiment_config.__dict__.items()
                             if not k.startswith('_')}
    }

    # Save to JSON
    save_json(combined_config, f"{output_dir}/config.json")

    return combined_config


def append_to_jsonl_file(file_path, new_data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Append the new data as a separate line
    with open(file_path, 'a') as f:
        f.write(json.dumps(new_data) + '\n')