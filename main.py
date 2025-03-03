import argparse
import os
import json
import numpy as np
from config.model_config import ModelConfig
from config.experiment_config import ExperimentConfig
from data.cot_datasets import load_gsm8k_dataset
from models.contemp_generator import ContemplationGenerator
from training.train_contemp_gen import train_contemplation_generator
from training.train_sent_trans import train_sentence_transformer, prepare_reasoning_pairs_dataset
from inference.inference import run_inference
from evaluation.metrics import evaluate_model
from evaluation.baselines import run_baseline
import utils.utils as utils
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser(description="Contemplation Tokens with Reasoning Ability")
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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--variation", type=str, default="vanilla", choices=["vanilla", "no_sentence_transformer", "no_l_reason"],
                        help="Variation of the effi_cot model to use")
    parser.add_argument("--baseline_type", type=str, default="vanilla_cot", choices=["vanilla_cot"],
                        help="Baseline type for evaluation")
    return parser.parse_args()

def run_experiment_sequence(variation, device, experiment_file, base_seed):
    """
    Run a sequence of experiments defined in a JSON file

    Args:
        experiment_file: Path to JSON file with experiment configurations
        base_seed: Base random seed to use
    """
    # Load experiment configurations
    with open(experiment_file, 'r') as f:
        experiments = json.load(f)

    # Create results directory
    results_dir = "./results"
    utils.create_directory(results_dir)

    # Prepare results collection
    all_results = []

    for i, exp in enumerate(experiments):
        print(f"\n{'='*80}")
        print(f"Running experiment {i+1}/{len(experiments)}: {exp.get('name', f'Experiment {i+1}')}")
        print(f"{'='*80}\n")

        # Extract experiment parameters
        exp_params = exp.get("parameters", {})
        repeat_count = exp.get("repeats", 1)

        # Prepare result collector for this experiment
        exp_results = {
            "name": exp.get("name", f"Experiment {i+1}"),
            "parameters": exp_params,
            "metrics": {}
        }

        # Collect metrics across repeats
        repeat_metrics = {}

        for repeat in range(repeat_count):
            # Set different seed for each repeat
            current_seed = base_seed + repeat
            utils.set_seed(current_seed)
            print(f"\nRepeat {repeat+1}/{repeat_count} with seed {current_seed}")

            # Create experiment-specific config
            model_config = ModelConfig("small")
            experiment_config = ExperimentConfig("default")
            experiment_config.device = device

            reasoning_pairs_path = os.path.join(experiment_config.reasoning_pairs_path,
                                                f"{model_config.teacher_model_name}/reasoning_pairs_{current_seed}.json")

            # Override with experiment parameters
            for key, value in exp_params.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
                elif hasattr(experiment_config, key):
                    setattr(experiment_config, key, value)

            # Load dataset
            train_dataset, eval_dataset = load_gsm8k_dataset(model_config.data_path)

            # Train sentence transformer
            queries = [item["query"] for item in train_dataset][:experiment_config.max_reasoning_pairs]
            if os.path.exists(reasoning_pairs_path):
                pairs_dataset = utils.load_json(reasoning_pairs_path)
            else:
                pairs_dataset = prepare_reasoning_pairs_dataset(
                    model_config.teacher_model_name,
                    device,
                    queries,
                    max_pairs=experiment_config.max_reasoning_pairs
                )
                # save to gen_datasets folder
                utils.create_directory(reasoning_pairs_path)
                utils.save_json(pairs_dataset, reasoning_pairs_path)

            # Unique output directory for this experiment
            exp_sent_trans_dir = f"{experiment_config.model_save_path}/sent_trans_{exp.get('name', '')}_repeat{repeat}"
            utils.create_directory(exp_sent_trans_dir)

            # Override save path
            experiment_config.model_save_path = exp_sent_trans_dir

            # Train sentence transformer
            sentence_transformer = train_sentence_transformer(
                model_config.teacher_model_name,
                experiment_config.start_layer_idx,
                experiment_config.end_layer_idx,
                pairs_dataset,
                experiment_config
            )

            # Train contemplation generator
            # Unique output directory for this experiment
            exp_contemp_gen_dir = f"{experiment_config.model_save_path}/contemp_gen_{exp.get('name', '')}_repeat{repeat}"
            utils.create_directory(exp_contemp_gen_dir)

            # Override save path
            experiment_config.model_save_path = exp_contemp_gen_dir

            # Initialize contemplation generator
            contemp_generator = ContemplationGenerator(
                model_config.student_model_name,
                model_config.teacher_model_name,
                model_config.teacher_hidden_dim,  # Pass the teacher's hidden dimension
                device=device
            )

            # Train the contemplation generator
            contemp_generator = train_contemplation_generator(
                contemp_generator,
                sentence_transformer,
                train_dataset,
                eval_dataset,
                experiment_config,
                variation
            )

            # Evaluate
            results = run_inference(
                contemp_generator,
                eval_dataset,
                model_config.teacher_model_name,
                experiment_config
            )

            # Calculate metrics
            metrics = evaluate_model(results, eval_dataset)

            # Store metrics for this repeat
            for metric_name, metric_value in metrics.items():
                if metric_name not in repeat_metrics:
                    repeat_metrics[metric_name] = []
                repeat_metrics[metric_name].append(metric_value)

            # Save individual repeat results
            repeat_results = {
                "seed": current_seed,
                "metrics": metrics
            }

            utils.save_json(
                repeat_results,
                f"{results_dir}/{exp.get('name', 'experiment')}_{repeat}.json"
            )

        # Calculate statistics across repeats
        for metric_name, values in repeat_metrics.items():
            values_array = np.array(values)
            exp_results["metrics"][metric_name] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "values": values
            }

        # Add experiment results to all results
        all_results.append(exp_results)

        # Save aggregated experiment results
        utils.save_json(
            exp_results,
            f"{results_dir}/{exp.get('name', 'experiment')}_aggregated.json"
        )

    # Save all experiment results
    utils.save_json(all_results, f"{results_dir}/all_experiments.json")

    # Print comparative summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    # Create a summary table
    header = ["Experiment", "Parameters"]

    # Find all unique metrics
    all_metrics = set()
    for exp_result in all_results:
        all_metrics.update(exp_result["metrics"].keys())

    # Add metrics to header
    for metric in sorted(all_metrics):
        header.append(f"{metric} (mean±std)")

    # Print header
    print(" | ".join(header))
    print("-" * (sum(len(h) for h in header) + 3 * (len(header) - 1)))

    # Print experiment rows
    for exp_result in all_results:
        row = [exp_result["name"]]

        # Add parameters
        param_str = ", ".join(f"{k}={v}" for k, v in exp_result["parameters"].items())
        row.append(param_str[:30] + "..." if len(param_str) > 30 else param_str)

        # Add metrics
        for metric in sorted(all_metrics):
            if metric in exp_result["metrics"]:
                mean = exp_result["metrics"][metric]["mean"]
                std = exp_result["metrics"][metric]["std"]
                row.append(f"{mean:.4f}±{std:.4f}")
            else:
                row.append("N/A")

        print(" | ".join(row))

def main():
    login(token='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')
    args = parse_args()


    # Set random seed
    utils.set_seed(args.seed)

    if args.mode == "run_experiments":
        run_experiment_sequence(args.device, args.experiment_file, args.seed)
    else:
        # Original logic for individual modes
        model_config = ModelConfig(args.config)
        # model_config.sentence_transformer_path = f"./saved_models/effi_cot/{args.baseline}/{args.variation}/sentence_transformer"

        experiment_config = ExperimentConfig(args.config)
        experiment_config.device = args.device
        experiment_config.model_save_path = f"{experiment_config.model_save_path}/{args.baseline}/{args.variation}" if args.baseline == 'effi_cot' else f"{experiment_config.model_save_path}/{args.baseline}"
        experiment_config.checkpoint_path = f"{experiment_config.checkpoint_path}/{args.baseline}/{args.variation}" if args.baseline == 'effi_cot' else f"{experiment_config.checkpoint_path}/{args.baseline}"
        experiment_config.result_path = f"{experiment_config.result_path}/{args.baseline}/{args.variation}" if args.baseline == 'effi_cot' else f"{experiment_config.result_path}/{args.baseline}"


        experiment_config.experiment_name = f"{args.baseline}_{args.variation}_{args.seed}"

        if not os.path.exists(experiment_config.model_save_path):
            os.makedirs(experiment_config.model_save_path)
        if not os.path.exists(experiment_config.checkpoint_path):
            os.makedirs(experiment_config.checkpoint_path)
        if not os.path.exists(experiment_config.result_path):
            os.makedirs(experiment_config.result_path)

        reasoning_pairs_path = os.path.join(experiment_config.reasoning_pairs_path,
                                                f"{model_config.teacher_model_name}/reasoning_pairs_{args.seed}.json")

        # Load dataset
        train_dataset, eval_dataset = load_gsm8k_dataset(model_config.data_path)

        if args.mode == "train_sentence_transformer" and args.variation == "vanilla":
            # Extract queries from the dataset
            queries = [item["query"] for item in train_dataset][:experiment_config.max_reasoning_pairs]

            # Prepare reasoning pairs dataset
            from training.train_sent_trans import prepare_reasoning_pairs_dataset
            if os.path.exists(reasoning_pairs_path):
                pairs_dataset = utils.load_json(reasoning_pairs_path)
            else:
                pairs_dataset = prepare_reasoning_pairs_dataset(
                    model_config.teacher_model_name,
                    args.device,
                    queries,
                    max_pairs=experiment_config.max_reasoning_pairs
                )
                # create directory
                utils.create_directory(reasoning_pairs_path)
                # save to gen_datasets folder
                utils.save_json(pairs_dataset, reasoning_pairs_path)



            # Train sentence transformer
            from training.train_sent_trans import train_sentence_transformer
            sentence_transformer = train_sentence_transformer(
                model_config.teacher_model_name,
                experiment_config.start_layer_idx,
                experiment_config.end_layer_idx,
                pairs_dataset,
                experiment_config
            )

        elif args.mode == "train_contemp_generator":
            # load condensed reasoning pairs dataset, add condensed reasoning to train_dataset
            pairs_dataset = utils.load_json(reasoning_pairs_path)
            # add to train dataset items with condensed reasoning of pairs_dataset
            for i in range(len(train_dataset)):
                train_dataset.update_item(i, "condensed_reasoning", pairs_dataset[i]["condensed_reasoning"])

            # Load pre-trained sentence transformer
            from models.sentence_transformer import CustomizedSentenceTransformer
            if args.variation != "vanilla":
                sentence_transformer = None
            else:
                sentence_transformer = CustomizedSentenceTransformer.from_pretrained(
                    experiment_config.model_save_path+"/sentence_transformer"
                )

            # Initialize contemplation generator
            contemp_generator = ContemplationGenerator(
                model_config.student_model_name,
                model_config.teacher_model_name,
                model_config.teacher_hidden_dim,
                device=args.device
            )

            # Train the contemplation generator
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
            # Load trained models and run inference
            contemp_generator = ContemplationGenerator.from_pretrained(
                experiment_config.model_save_path+"/contemp_generator"
            )
            results = run_inference(
                contemp_generator,
                eval_dataset,
                model_config.teacher_model_name,
                experiment_config
            )
            # Evaluate results
            metrics = evaluate_model(results, eval_dataset)
            # save results
            utils.save_json(metrics, f"{experiment_config.result_path}/evaluation_results.json")
            print(f"Evaluation results: {metrics}")

        elif args.mode == "baseline":
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

if __name__ == "__main__":
    main()