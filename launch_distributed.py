#!/usr/bin/env python
import os
import torch
import argparse
import json
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from config.model_config import ModelConfig
from config.experiment_config import ExperimentConfig
from data.cot_datasets import load_gsm8k_dataset
from models.contemp_generator import ContemplationGenerator
from models.sentence_transformer import CustomizedSentenceTransformer
from training.distributed_train_sent_trans import train_sentence_transformer_distributed, prepare_reasoning_pairs_dataset_distributed
from training.distributed_train_contemp_gen import train_contemplation_generator_distributed
from inference.inference import run_inference
from evaluation.metrics import evaluate_model
from evaluation.baselines import run_baseline
import utils.utils as utils
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Contemplation Tokens Training")
    parser.add_argument("--mode", type=str,
                      choices=["train_sentence_transformer", "train_contemp_generator",
                               "evaluate", "baseline", "run_experiments"],
                      default="train_sentence_transformer",
                      help="Operation mode")
    parser.add_argument("--config", type=str, default="small",
                      help="Configuration name")
    parser.add_argument("--baseline", type=str, default="effi_cot",
                      choices=["effi_cot", "ccot", "pause", "implicit_cot"],
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

    # Add distributed specific arguments (will be set by torchrun)
    parser.add_argument('--local_rank', type=int, default=0,
                      help='Local rank for distributed training (set by torchrun)')

    return parser.parse_args()

def run_distributed_training(local_rank, world_size):
    """Main function for distributed training"""
    # Parse arguments
    args = parse_args()

    # Setup distributed environment
    setup_distributed(local_rank, world_size)

    # Set device
    device = local_rank

    # Login to HuggingFace (only on main process)
    if is_main_process(local_rank):
        login(token='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')

    # Set seed
    utils.set_seed(args.seed + local_rank)  # Add rank to seed for diversity

    try:
        # Load configs
        model_config = ModelConfig(args.config)
        experiment_config = ExperimentConfig(args.config)
        experiment_config.device = device

        # Setup experiment paths
        experiment_config.model_save_path = f"{experiment_config.model_save_path}/{args.baseline}/{args.variation}" if args.baseline == 'effi_cot' else f"{experiment_config.model_save_path}/{args.baseline}"
        experiment_config.checkpoint_path = f"{experiment_config.checkpoint_path}/{args.baseline}/{args.variation}" if args.baseline == 'effi_cot' else f"{experiment_config.checkpoint_path}/{args.baseline}"
        experiment_config.result_path = f"{experiment_config.result_path}/{args.baseline}/{args.variation}" if args.baseline == 'effi_cot' else f"{experiment_config.result_path}/{args.baseline}"
        experiment_config.experiment_name = f"{args.baseline}_{args.variation}_{args.seed}"

        # Create directories (only on main process)
        if is_main_process(local_rank):
            utils.create_directory(experiment_config.model_save_path)
            utils.create_directory(experiment_config.checkpoint_path)
            utils.create_directory(experiment_config.result_path)

        # Load dataset
        train_dataset, eval_dataset = load_gsm8k_dataset(model_config.data_path)

        # Make sure all processes have loaded the dataset
        torch.distributed.barrier()

        # Reasoning pairs path
        reasoning_pairs_path = os.path.join(experiment_config.reasoning_pairs_path,
                                          f"{model_config.teacher_model_name}/reasoning_pairs_{args.seed}.json")

        if args.mode == "train_sentence_transformer" and args.variation == "vanilla":
            # Extract queries from the dataset
            queries = [item["query"] for item in train_dataset][:experiment_config.max_reasoning_pairs]

            # Prepare reasoning pairs dataset (only on rank 0)
            if is_main_process(local_rank):
                if os.path.exists(reasoning_pairs_path):
                    print(f"Loading existing reasoning pairs from {reasoning_pairs_path}")
                    pairs_dataset = utils.load_json(reasoning_pairs_path)
                else:
                    print(f"Generating new reasoning pairs (this may take some time)...")
                    # Explicitly clear GPU memory before generating pairs
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                    pairs_dataset = prepare_reasoning_pairs_dataset_distributed(
                        model_config.teacher_model_name,
                        queries,
                        max_pairs=experiment_config.max_reasoning_pairs,
                        rank=local_rank
                    )
                    # Create directory and save to gen_datasets folder
                    utils.create_directory(os.path.dirname(reasoning_pairs_path))
                    utils.save_json(pairs_dataset, reasoning_pairs_path)
                    print(f"Saved reasoning pairs to {reasoning_pairs_path}")

            # Ensure all processes wait for dataset creation
            torch.distributed.barrier()

            # Now all processes load the dataset
            pairs_dataset = utils.load_json(reasoning_pairs_path)

            # Train sentence transformer with distributed training
            sentence_transformer = train_sentence_transformer_distributed(
                model_config.teacher_model_name,
                experiment_config.start_layer_idx,
                experiment_config.end_layer_idx,
                pairs_dataset,
                experiment_config,
                local_rank,
                world_size
            )

        elif args.mode == "train_contemp_generator":
            # Load condensed reasoning pairs dataset
            pairs_dataset = utils.load_json(reasoning_pairs_path)

            # Add to train dataset items with condensed reasoning of pairs_dataset
            for i in range(len(train_dataset)):
                train_dataset.update_item(i, "condensed_reasoning", pairs_dataset[i]["condensed_reasoning"])

            # Load pre-trained sentence transformer
            if args.variation == "no_sentence_transformer":
                sentence_transformer = None
            else:
                sentence_transformer = CustomizedSentenceTransformer.from_pretrained(
                    model_config.sentence_transformer_path
                )

            # Initialize contemplation generator
            contemp_generator = ContemplationGenerator(
                model_config.student_model_name,
                model_config.teacher_model_name,
                model_config.teacher_hidden_dim,
                device=device
            )

            # Train with distributed training
            train_contemplation_generator_distributed(
                contemp_generator,
                sentence_transformer,
                train_dataset,
                eval_dataset,
                model_config,
                experiment_config,
                args.variation,
                local_rank,
                world_size
            )

        elif args.mode == "evaluate":
            # This mode typically runs inference, which might not benefit from DDP
            # But we ensure only the main process runs it
            if is_main_process(local_rank):
                # Load trained models and run inference
                contemp_generator = ContemplationGenerator.from_pretrained(
                    model_config.student_model_path
                )
                results = run_inference(
                    contemp_generator,
                    eval_dataset,
                    model_config.teacher_model_name,
                    experiment_config
                )
                # Evaluate results
                metrics = evaluate_model(results, eval_dataset)
                print(f"Evaluation results: {metrics}")

        elif args.mode == "baseline":
            # Similar to evaluate, baselines typically don't benefit from DDP
            if is_main_process(local_rank):
                # Run baseline method
                results = run_baseline(
                    args.baseline,
                    eval_dataset,
                    model_config,
                    experiment_config
                )
                # Evaluate results
                metrics = evaluate_model(results, eval_dataset)
                print(f"Baseline {args.baseline} results: {metrics}")

    finally:
        # Clean up distributed environment
        cleanup_distributed()


def main():
    # Initialize distributed environment
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Run distributed training
    run_distributed_training(local_rank, world_size)


if __name__ == "__main__":
    main()