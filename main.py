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
                        choices=["cot", "ccot", "pause", "icot_kd", "zero_shot_cot", "effi_cot", "icot_si", "codi", "softcot"],
                        help="Baseline to run if mode is baseline")
    parser.add_argument("--ccot_stage", type=str, default="encode",choices=["encode", "decode", "prepare_decode_data", "evaluate", "cotrain_encode_decode"],
                        help="Stage for CCoT")
    # Add CODI stage for CODI baseline
    parser.add_argument("--codi_stage", type=str, default="train",
                                choices=["train", "evaluate"], help="Stage for CODI baseline")

    parser.add_argument("--experiment_file", type=str, default="experiments.json",
                        help="JSON file containing experiment configurations")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--variation", type=str, default="vanilla", choices=["vanilla", "no_sentence_transformer", "no_l_reason"],
                        help="Variation of the effi_cot model to use")
    parser.add_argument("--compression_ratio", type=float, default=0.05,
                        help="Compression ratio for CCoT (ratio of compressed tokens to full chain)")
    parser.add_argument("--train_max_contemp_tokens", type=int, default=5, help="max number of contemp tokens for training efficot, also served as maximum number of contemplation tokens for CCoT (this conflicts with compression ratio, just for temporary debug, should be removed very soon)")
    parser.add_argument("--eval_max_contemp_tokens", type=int, default=1, help="max number of contemp tokens for evaluating efficot, also served as maximum number of contemplation tokens for CCoT (this conflicts with compression ratio, just for temporary debug, should be removed very soon)")
    parser.add_argument("--autoregressive_layer", type=int, default=15,
                        help="Layer to use for autoregressive generation in CCoT")
    parser.add_argument("--cot_bsl_shot", type=int, default=0,
                        help="Number of shots for cot baseline")
    parser.add_argument("--eval_temp", type=float, default=0.7,
                        help="Temperature for evaluation")


    parser.add_argument("--sent_trans_lr", type=float, default=1e-5,
                    help="Learning rate for sentence transformer")
    parser.add_argument("--sent_trans_weight_decay", type=float, default=0.01,
                        help="Weight decay for sentence transformer")
    parser.add_argument("--sent_trans_epochs", type=int, default=15,
                        help="Number of epochs for sentence transformer training")

    parser.add_argument("--contemp_gen_lr", type=float, default=1e-7,
                        help="Learning rate for contemporary generator")
    parser.add_argument("--contemp_gen_weight_decay", type=float, default=1e-5,
                        help="Weight decay for contemporary generator")
    parser.add_argument("--contemp_gen_epochs", type=int, default=2,
                        help="Number of epochs for contemporary generator training")

    parser.add_argument("--contemp_gen_lin_layer_lr", type=float, default=0.001,
                        help="Learning rate for contemporary generator linear layer")
    parser.add_argument("--contemp_gen_lin_layer_weight_decay", type=float, default=0.001,
                        help="Weight decay for contemporary generator linear layer")
    parser.add_argument("--contemp_gen_lin_layer_epochs", type=int, default=10,
                        help="Number of epochs for contemporary generator linear layer training")

    return parser.parse_args()


def main():
    os.environ['HF_HOME'] = '/data/nee7ne/huggingface'
    # login(token='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')
    args = parse_args()
    # Set random seed
    utils.set_seed(args.seed)
    # Original logic for individual modes
    model_config = ModelConfig(args.config)
    experiment_config = ExperimentConfig(args.config)
    experiment_config.device = args.device
    experiment_config.ccot_stage = args.ccot_stage
    experiment_config.train_max_contemp_tokens = args.train_max_contemp_tokens if args.train_max_contemp_tokens is not None else experiment_config.train_max_contemp_tokens # SHOULD BE REMOVED, JUST DEBUG FOR DIFFERENT CONTEMP FOR CCOT
    experiment_config.eval_max_contemp_tokens = args.eval_max_contemp_tokens if args.eval_max_contemp_tokens is not None else experiment_config.eval_max_contemp_tokens # SHOULD BE REMOVED, JUST DEBUG FOR DIFFERENT CONTEMP FOR CCOT
    experiment_config.eval_temp = args.eval_temp

    # reset lr and wd and epochs of experiment config
    experiment_config.sent_trans_lr = args.sent_trans_lr
    experiment_config.sent_trans_weight_decay = args.sent_trans_weight_decay
    experiment_config.sent_trans_epochs = args.sent_trans_epochs

    experiment_config.contemp_gen_lr = args.contemp_gen_lr
    experiment_config.contemp_gen_epochs = args.contemp_gen_epochs
    experiment_config.contemp_gen_lin_layer_lr = args.contemp_gen_lin_layer_lr
    experiment_config.contemp_gen_lin_layer_epochs = args.contemp_gen_lin_layer_epochs

    experiment_config.contemp_gen_lin_layer_weight_decay = args.contemp_gen_lin_layer_weight_decay
    experiment_config.contemp_gen_weight_decay = args.contemp_gen_weight_decay

    experiment_config.train_max_contemp_tokens = args.train_max_contemp_tokens
    experiment_config.eval_max_contemp_tokens = args.eval_max_contemp_tokens

    # Special handling for CCoT mode
    if args.mode == "train_ccot" or (args.mode == "baseline" and args.baseline == "ccot"):
        experiment_config.model_save_path = f"{experiment_config.model_save_path}/ccot/{args.config}/{args.dataset}"
        experiment_config.checkpoint_path = f"{experiment_config.checkpoint_path}/ccot/{args.config}/{args.dataset}"
        experiment_config.result_path = f"{experiment_config.result_path}/ccot/{args.config}/{args.dataset}"
        experiment_config.experiment_name = f"ccot_{args.compression_ratio}_{args.seed}_{args.dataset}_{args.config}"

        # Add compression ratio and autoregressive layer to experiment config
        experiment_config.compression_ratio = args.compression_ratio
        experiment_config.autoregressive_layer = args.autoregressive_layer
    elif args.mode == "baseline" and args.baseline == "codi":
        experiment_config.model_save_path = f"{experiment_config.model_save_path}/codi/{args.config}/{args.dataset}"
        experiment_config.checkpoint_path = f"{experiment_config.checkpoint_path}/codi/{args.config}/{args.dataset}"
        experiment_config.result_path = f"{experiment_config.result_path}/codi/{args.config}/{args.dataset}"
        experiment_config.experiment_name = f"codi_{args.seed}_{args.dataset}_{args.config}"

        # Add CODI stage to experiment config
        experiment_config.codi_stage = args.codi_stage
    elif args.mode == "baseline" and args.baseline == "icot_kd":
        experiment_config.model_save_path = f"{experiment_config.model_save_path}/icot_kd/{args.config}/{args.dataset}"
        experiment_config.checkpoint_path = f"{experiment_config.checkpoint_path}/icot_kd/{args.config}/{args.dataset}"
        experiment_config.result_path = f"{experiment_config.result_path}/icot_kd/{args.config}/{args.dataset}"
        experiment_config.experiment_name = f"icot_kd_{args.seed}_{args.dataset}_{args.config}"
    else:
        # Original path handling for other modes
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
            experiment_config.model_save_path+"/contemp_generator/"+model_config.student_model_name+"/"
        )
        results = run_inference(
            contemp_generator,
            eval_dataset,
            model_config.teacher_model_name,
            experiment_config
        )
        # Evaluate results
        metrics = evaluate_model(results, eval_dataset)

        metrics.update({
        'dataset': args.dataset,
        'student': model_config.student_model_name,
        'teacher': model_config.teacher_model_name,
        'sent_trans_lr': experiment_config.sent_trans_lr,
        'sent_trans_weight_decay': experiment_config.sent_trans_weight_decay,
        'sent_trans_epochs': experiment_config.sent_trans_epochs,
        'contemp_gen_lr': experiment_config.contemp_gen_lr,
        'contemp_gen_epochs': experiment_config.contemp_gen_epochs,
        'contemp_gen_lin_layer_lr': experiment_config.contemp_gen_lin_layer_lr,
        'contemp_gen_lin_layer_epochs': experiment_config.contemp_gen_lin_layer_epochs,
        'contemp_gen_lin_layer_weight_decay': experiment_config.contemp_gen_lin_layer_weight_decay,
        'contemp_gen_weight_decay': experiment_config.contemp_gen_weight_decay,
        'eval_temp': experiment_config.eval_temp,
        'train_max_contemp_tokens': experiment_config.train_max_contemp_tokens,
        'eval_max_contemp_tokens': experiment_config.eval_max_contemp_tokens,
        })

        # save results
        # utils.save_json(metrics, f"{experiment_config.result_path}/evaluation_results.json")
        utils.append_to_jsonl_file(f"{experiment_config.result_path}/evaluation_results.jsonl", metrics)
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
        metrics.update({
            'dataset': args.dataset,
            'eval_temp': experiment_config.eval_temp,
            'eval_max_contemp_tokens': experiment_config.eval_max_contemp_tokens,
        })
        # save
        utils.append_to_jsonl_file(f"{experiment_config.result_path}/evaluation_results.jsonl", metrics)
        print(f"Baseline {args.baseline} results: {metrics}")
if __name__ == "__main__":
    main()