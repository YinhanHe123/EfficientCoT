import argparse
import os
import json
import numpy as np
from config.model_config import ModelConfig
from config.experiment_config import ExperimentConfig
from data.cot_datasets import load_raw_dataset
from models.contemp_generator import ContemplationGenerator
from training.train_contemp_gen import train_contemplation_generator
from training.train_sent_trans import train_sentence_transformer, prepare_reasoning_pairs_dataset
from inference.inference import run_inference
from evaluation.metrics import evaluate_model
from baselines.baselines import run_baseline
import utils.utils as utils
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser(description="Contemplation Tokens with Reasoning Ability")
    parser.add_argument("--mode", type=str,
                        choices=["train_sentence_transformer", "train_contemp_generator",
                                 "evaluate", "baseline", "run_experiments", "train_ccot"],
                        default="train_sentence_transformer",
                        help="Operation mode")
    parser.add_argument("--config", type=str, default="small",
                        help="Configuration name")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "svamp", "multiarith"],
                        help="Dataset to use")
    parser.add_argument("--baseline", type=str, default="effi_cot",
                        choices=["cot", "ccot", "pause", "implicit_cot","zero_shot_cot", "effi_cot"],
                        help="Baseline to run if mode is baseline")
    parser.add_argument("--ccot_stage", type=str, default="encode",choices=["encode", "decode", "prepare_decode_data", "evaluate", "cotrain_encode_decode"],
                        help="Stage for CCoT")
    parser.add_argument("--experiment_file", type=str, default="experiments.json",
                        help="JSON file containing experiment configurations")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--variation", type=str, default="vanilla", choices=["vanilla", "no_sentence_transformer", "no_l_reason"],
                        help="Variation of the effi_cot model to use")
    parser.add_argument("--compression_ratio", type=float, default=0.1,
                        help="Compression ratio for CCoT (ratio of compressed tokens to full chain)")
    parser.add_argument("--max_contemp_tokens", type=int, default=None, help="Maximum number of contemplation tokens for CCoT (this conflicts with compression ratio, just for temporary debug, should be removed very soon)")
    parser.add_argument("--autoregressive_layer", type=int, default=15,
                        help="Layer to use for autoregressive generation in CCoT")
    parser.add_argument("--cot_bsl_shot", type=int, default=0,
                        help="Number of shots for cot baseline")
    parser.add_argument("--eval_temp", type=float, default=0.3,
                        help="Temperature for evaluation")

    return parser.parse_args()

def main():
    os.environ['HF_HOME'] = '/data/nee7ne/huggingface'
    login(token='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')
    args = parse_args()
    # Set random seed
    utils.set_seed(args.seed)
    # Original logic for individual modes
    model_config = ModelConfig(args.config)
    experiment_config = ExperimentConfig(args.config)
    experiment_config.device = args.device
    experiment_config.ccot_stage = args.ccot_stage
    experiment_config.max_contemp_tokens = args.max_contemp_tokens if args.max_contemp_tokens is not None else experiment_config.max_contemp_tokens # SHOULD BE REMOVED, JUST DEBUG FOR DIFFERENT CONTEMP FOR CCOT
    experiment_config.eval_temp = args.eval_temp

    # Special handling for CCoT mode
    if args.mode == "train_ccot" or (args.mode == "baseline" and args.baseline == "ccot"):
        experiment_config.model_save_path = f"{experiment_config.model_save_path}/ccot/{args.dataset}"
        experiment_config.checkpoint_path = f"{experiment_config.checkpoint_path}/ccot/{args.dataset}"
        experiment_config.result_path = f"{experiment_config.result_path}/ccot/{args.dataset}"
        experiment_config.experiment_name = f"ccot_{args.compression_ratio}_{args.seed}"

        # Add compression ratio and autoregressive layer to experiment config
        experiment_config.compression_ratio = args.compression_ratio
        experiment_config.autoregressive_layer = args.autoregressive_layer
    else:
        # Original path handling for other modes
        # experiment_config.model_save_path = f"{experiment_config.model_save_path}/{args.baseline}/{args.variation}/{args.dataset}" if args.baseline == 'effi_cot' else f"{experiment_config.model_save_path}/{args.baseline}/{args.dataset}"
        # experiment_config.checkpoint_path = f"{experiment_config.checkpoint_path}/{args.baseline}/{args.variation}/{args.dataset}" if args.baseline == 'effi_cot' else f"{experiment_config.checkpoint_path}/{args.baseline}/{args.dataset}"
        # experiment_config.result_path = f"{experiment_config.result_path}/{args.baseline}/{args.variation}/{args.dataset}" if args.baseline == 'effi_cot' else f"{experiment_config.result_path}/{args.baseline}/{args.dataset}"
        # experiment_config.experiment_name = f"{args.baseline}_{args.variation}_{args.seed}_{args.dataset}"
        experiment_config.model_save_path = f"{experiment_config.model_save_path}/{args.baseline}/{args.variation}/{args.config}/{args.dataset}" if args.baseline == 'effi_cot' else f"{experiment_config.model_save_path}/{args.baseline}/{args.config}/{args.dataset}"
        experiment_config.checkpoint_path = f"{experiment_config.checkpoint_path}/{args.baseline}/{args.variation}/{args.config}/{args.dataset}" if args.baseline == 'effi_cot' else f"{experiment_config.checkpoint_path}/{args.baseline}/{args.config}/{args.dataset}"
        experiment_config.result_path = f"{experiment_config.result_path}/{args.baseline}/{args.variation}/{args.config}/{args.dataset}" if args.baseline == 'effi_cot' else f"{experiment_config.result_path}/{args.baseline}/{args.config}/{args.dataset}"
        experiment_config.experiment_name = f"{args.baseline}_{args.variation}_{args.seed}_{args.dataset}_{args.config}"

    # Create necessary directories
    if not os.path.exists(experiment_config.model_save_path):
        os.makedirs(experiment_config.model_save_path)
    if not os.path.exists(experiment_config.checkpoint_path):
        os.makedirs(experiment_config.checkpoint_path)
    if not os.path.exists(experiment_config.result_path):
        os.makedirs(experiment_config.result_path)

    reasoning_pairs_path = os.path.join(experiment_config.reasoning_pairs_path,
                                            f"{model_config.teacher_model_name}/{args.dataset}/reasoning_pairs_{args.seed}.json")

    # Load dataset
    if args.dataset == 'gsm8k':
        model_config.data_path = 'openai/gsm8k'
    elif args.dataset == 'svamp':
        model_config.data_path = 'ChilleD/SVAMP'
    elif args.dataset == 'multiarith':
        model_config.data_path = 'ChilleD/MultiArith'

    train_dataset, eval_dataset = load_raw_dataset(model_config.data_path)

    # Process different modes
    if args.mode == "train_sentence_transformer" and args.variation == "vanilla":
        # Extract queries from the dataset
        queries = [item["query"] for item in train_dataset][:experiment_config.max_reasoning_pairs]
        if "full_answer" in train_dataset[0].keys() and train_dataset[0]["full_answer"] != "":
            reasonings = [item["full_answer"] for item in train_dataset][:experiment_config.max_reasoning_pairs]
        else:
            reasonings = None
        answers = [item["answer"] for item in train_dataset][:experiment_config.max_reasoning_pairs]
        # Prepare reasoning pairs dataset
        from training.train_sent_trans import prepare_reasoning_pairs_dataset
        if os.path.exists(reasoning_pairs_path):
            pairs_dataset = utils.load_json(reasoning_pairs_path)
        else:
            pairs_dataset = prepare_reasoning_pairs_dataset(
                queries,
                reasonings,
                answers,
                reasoning_pairs_path,
                max_pairs=experiment_config.max_reasoning_pairs
            )


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
            train_dataset,
            eval_dataset,
            model_config,
            experiment_config,
            num_shots=args.cot_bsl_shot
        )
        # Evaluate results
        metrics = evaluate_model(results, eval_dataset)
        # save
        utils.save_json(metrics, f"{experiment_config.result_path}/{args.cot_bsl_shot}_shot__baseline_results.json")
        print(f"Baseline {args.baseline} results: {metrics}")
if __name__ == "__main__":
    main()