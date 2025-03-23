import argparse
import os
import json
import numpy as np
from config.model_config import ModelConfig
from config.experiment_config import ExperimentConfig
from data.cot_datasets import load_raw_dataset
from models.deepspeed_contemp_generator import PipelinedContemplationGenerator
from models.deepspeed_sentence_transformer import PipelinedSentenceTransformer
from training.deepspeed_train_sent_trans import train_sentence_transformer_with_deepspeed
from training.deepspeed_train_contemp_gen import train_contemplation_generator_with_deepspeed
from inference.deepspeed_inference import run_inference_with_deepspeed
from evaluation.metrics import evaluate_model
from baselines.baselines import run_baseline
import utils.utils as utils
from huggingface_hub import login
import deepspeed
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Enhanced EfficientCoT")
    parser.add_argument("--mode", type=str,
                        choices=["train_sentence_transformer", "train_contemp_generator",
                                 "evaluate", "baseline", "run_experiments"],
                        default="train_sentence_transformer",
                        help="Operation mode")
    parser.add_argument("--config", type=str, default="small",
                        choices=["small", "large"],
                        help="Configuration name ('small' for 7B/1.3B models, 'large' for 13B/7B models)")
    parser.add_argument("--baseline", type=str, default="effi_cot",
                        choices=["cot", "ccot", "pause", "implicit_cot","zero_shot_cot"],
                        help="Baseline to run if mode is baseline")
    parser.add_argument("--experiment_file", type=str, default="experiments.json",
                        help="JSON file containing experiment configurations")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--variation", type=str, default="vanilla",
                        choices=["vanilla", "no_sentence_transformer", "no_l_reason"],
                        help="Variation of the effi_cot model to use")
    parser.add_argument("--num_stages", type=int, default=2,
                        help="Number of pipeline stages for DeepSpeed")
    parser.add_argument("--cot_bsl_shot", type=int, default=0,
                        help="Number of shots for cot baseline")
    parser.add_argument("--use_deepspeed", action="store_true",
                        help="Use DeepSpeed for training/inference")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Set up distributed training if using DeepSpeed
    if args.use_deepspeed and args.local_rank == -1:
        print("Warning: Using DeepSpeed but local_rank is -1. This might cause issues.")
        args.local_rank = 0

    # Set random seed for reproducibility
    utils.set_seed(args.seed)

    # Initialize distributed environment if using DeepSpeed
    if args.use_deepspeed and not torch.distributed.is_initialized():
        deepspeed.init_distributed()

    # Load configurations
    model_config = ModelConfig(args.config)
    experiment_config = ExperimentConfig(args.config)
    experiment_config.device = f"cuda:{args.local_rank}" if args.use_deepspeed else args.local_rank

    # Set up directory paths based on configuration
    baseline_path = f"{args.baseline}/{args.variation}" if args.baseline == 'effi_cot' else f"{args.baseline}"
    experiment_config.model_save_path = f"{experiment_config.model_save_path}/{baseline_path}"
    experiment_config.checkpoint_path = f"{experiment_config.checkpoint_path}/{baseline_path}"
    experiment_config.result_path = f"{experiment_config.result_path}/{baseline_path}"
    experiment_config.experiment_name = f"{args.baseline}_{args.variation}_{args.seed}"

    # Create directories if they don't exist
    # Only have rank 0 create directories for distributed setups
    if not args.use_deepspeed or (args.use_deepspeed and args.local_rank == 0):
        for path in [experiment_config.model_save_path, experiment_config.checkpoint_path, experiment_config.result_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    # Set path for reasoning pairs
    reasoning_pairs_path = os.path.join(experiment_config.reasoning_pairs_path,
                                        f"{model_config.teacher_model_name}/reasoning_pairs_{args.seed}.json")

    # Load dataset
    train_dataset, eval_dataset = load_raw_dataset(model_config.data_path)

    # Login to Hugging Face if needed for downloading models
    login(token='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')

    # Execute the requested mode
    if args.mode == "train_sentence_transformer" and args.variation == "vanilla":
        # Extract queries from the dataset
        queries = [item["query"] for item in train_dataset][:experiment_config.max_reasoning_pairs]

        # Check if reasoning pairs already exist or need to be created
        if os.path.exists(reasoning_pairs_path):
            pairs_dataset = utils.load_json(reasoning_pairs_path)
        else:
            # Generate reasoning pairs
            from training.train_sent_trans import prepare_reasoning_pairs_dataset
            pairs_dataset = prepare_reasoning_pairs_dataset(
                model_config.teacher_model_name,
                experiment_config.device,
                queries,
                max_pairs=experiment_config.max_reasoning_pairs
            )

            # Save generated pairs
            if not args.use_deepspeed or (args.use_deepspeed and args.local_rank == 0):
                utils.create_directory(os.path.dirname(reasoning_pairs_path))
                utils.save_json(pairs_dataset, reasoning_pairs_path)

        # Train sentence transformer
        if args.use_deepspeed:
            train_sentence_transformer_with_deepspeed(
                model_config.teacher_model_name,
                experiment_config.start_layer_idx,
                experiment_config.end_layer_idx,
                pairs_dataset,
                experiment_config,
                args.local_rank,
                num_stages=args.num_stages
            )
        else:
            from training.train_sent_trans import train_sentence_transformer
            train_sentence_transformer(
                model_config.teacher_model_name,
                experiment_config.start_layer_idx,
                experiment_config.end_layer_idx,
                pairs_dataset,
                experiment_config
            )

    elif args.mode == "train_contemp_generator":
        # Load the reasoning pairs dataset
        pairs_dataset = utils.load_json(reasoning_pairs_path)

        # Add condensed reasoning to train dataset
        for i in range(min(len(train_dataset), len(pairs_dataset))):
            train_dataset.update_item(i, "condensed_reasoning", pairs_dataset[i]["condensed_reasoning"])

        # Load pre-trained sentence transformer based on variation
        if args.variation != "vanilla":
            sentence_transformer = None
        else:
            if args.use_deepspeed:
                sentence_transformer = PipelinedSentenceTransformer.from_pretrained(
                    f"{experiment_config.model_save_path}/sentence_transformer"
                )
            else:
                from models.sentence_transformer import CustomizedSentenceTransformer
                sentence_transformer = CustomizedSentenceTransformer.from_pretrained(
                    f"{experiment_config.model_save_path}/sentence_transformer"
                )

        # Initialize contemplation generator
        if args.use_deepspeed:
            contemp_generator = PipelinedContemplationGenerator(
                model_config.student_model_name,
                model_config.teacher_model_name,
                model_config.teacher_hidden_dim,
                device=experiment_config.device
            )

            # Train with DeepSpeed
            train_contemplation_generator_with_deepspeed(
                model_config,
                experiment_config,
                train_dataset,
                eval_dataset,
                args.local_rank,
                args.variation,
                num_stages=args.num_stages
            )
        else:
            from models.contemp_generator import ContemplationGenerator
            from training.train_contemp_gen import train_contemplation_generator

            contemp_generator = ContemplationGenerator(
                model_config.student_model_name,
                model_config.teacher_model_name,
                model_config.teacher_hidden_dim,
                device=experiment_config.device
            )

            # Train with regular PyTorch
            train_contemplation_generator(
                contemp_generator,
                sentence_transformer,
                train_dataset,
                eval_dataset,
                model_config,
                experiment_config,
                args.variation
            )

    elif args.mode == "evaluate":
        # Load the trained model for evaluation
        if args.use_deepspeed:
            model_path = f"{experiment_config.model_save_path}/contemp_generator"
            contemp_generator = PipelinedContemplationGenerator.from_pretrained(model_path)

            # Run inference with DeepSpeed
            results = run_inference_with_deepspeed(
                contemp_generator,
                eval_dataset,
                model_config.teacher_model_name,
                experiment_config,
                args.local_rank
            )
        else:
            from models.contemp_generator import ContemplationGenerator
            from inference.inference import run_inference

            contemp_generator = ContemplationGenerator.from_pretrained(
                f"{experiment_config.model_save_path}/contemp_generator"
            )

            # Run inference without DeepSpeed
            results = run_inference(
                contemp_generator,
                eval_dataset,
                model_config.teacher_model_name,
                experiment_config
            )

        # Evaluate results - only on rank 0 for distributed setups
        if not args.use_deepspeed or (args.use_deepspeed and args.local_rank == 0):
            metrics = evaluate_model(results, eval_dataset)
            utils.save_json(metrics, f"{experiment_config.result_path}/evaluation_results.json")
            print(f"Evaluation results: {metrics}")

    elif args.mode == "baseline":
        # Run baseline methods (these don't use DeepSpeed yet)
        results = run_baseline(
            args.baseline,
            eval_dataset,
            model_config,
            experiment_config,
            num_shots=args.cot_bsl_shot
        )

        # Evaluate baseline results
        metrics = evaluate_model(results, eval_dataset)
        utils.save_json(metrics, f"{experiment_config.result_path}/{args.cot_bsl_shot}_shot__baseline_results.json")
        print(f"Baseline {args.baseline} results: {metrics}")

    elif args.mode == "run_experiments":
        # Running full experiment suite is not supported with DeepSpeed yet
        if args.use_deepspeed:
            print("Running experiments with DeepSpeed is not supported yet. Please run individual modes instead.")
        else:
            from main import run_experiment_sequence
            run_experiment_sequence(args.variation, args.local_rank, args.experiment_file, args.seed)

    # Cleanup distributed environment if needed
    if args.use_deepspeed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()