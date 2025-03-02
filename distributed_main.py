import os
import argparse
import torch
import torch.multiprocessing as mp
from main import parse_args, run_experiment_sequence
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process
import utils.utils as utils
from huggingface_hub import login

def parse_distributed_args():
    """Parse arguments for distributed training"""
    parser = argparse.ArgumentParser(description="Distributed Contemplation Tokens with Reasoning Ability")
    parser.add_argument("--mode", type=str,
                        choices=["train_sentence_transformer", "train_contemp_generator",
                                 "evaluate", "baseline", "run_experiments"],
                        default="train_sentence_transformer",
                        help="Operation mode")
    parser.add_argument("--config", type=str, default="small",
                        help="Configuration name")
    parser.add_argument("--baseline", type=str, default="effi_cot",
                        choices=["ccot", "pause", "implicit_cot"],
                        help="Baseline to run if mode is baseline")
    parser.add_argument("--experiment_file", type=str, default="experiments.json",
                        help="JSON file containing experiment configurations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--variation", type=str, default="vanilla",
                        choices=["vanilla", "no_sentence_transformer", "no_l_reason"],
                        help="Variation of the effi_cot model to use")
    parser.add_argument("--baseline_type", type=str, default="vanilla_cot",
                        choices=["vanilla_cot"],
                        help="Baseline type for evaluation")

    # Add distributed training specific arguments
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(),
                        help="Number of GPUs to use for distributed training")
    parser.add_argument("--dist_url", default="env://", type=str,
                        help="URL used to set up distributed training")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training")

    return parser.parse_args()

def run_distributed_experiments(rank, world_size, args):
    """
    Run experiments in distributed mode

    Args:
        rank: Current process rank
        world_size: Total number of processes
        args: Command line arguments
    """
    # Setup distributed environment
    setup_distributed(rank, world_size)

    # Set device to the current GPU
    args.device = rank

    # Only log in to HuggingFace on the main process
    if is_main_process(rank):
        login(token='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')

    # Set random seed
    utils.set_seed(args.seed + rank)  # Add rank to seed for diversity

    try:
        # Run experiment sequence
        if args.mode == "run_experiments":
            run_experiment_sequence(args.variation, rank, args.experiment_file, args.seed + rank)
        else:
            # Original logic adapted for distributed training
            # Implement distributed mode for main.py original logic here
            pass
    finally:
        # Clean up distributed environment
        cleanup_distributed()

def main():
    args = parse_distributed_args()

    if args.distributed:
        # Use PyTorch's multiprocessing to spawn processes
        world_size = args.world_size
        mp.spawn(
            run_distributed_experiments,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Fall back to original non-distributed implementation
        # Set device to 0 (first GPU)
        args.device = 0

        # Login to HuggingFace
        login(token='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')

        # Set random seed
        utils.set_seed(args.seed)

        if args.mode == "run_experiments":
            run_experiment_sequence(args.variation, args.device, args.experiment_file, args.seed)
        else:
            # Original logic from main.py
            pass

if __name__ == "__main__":
    main()