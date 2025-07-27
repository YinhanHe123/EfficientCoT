import argparse
import os

import torch
os.environ['HF_HOME'] = '/data/nee7ne/huggingface'
from models.sentence_transformer import CustomizedSentenceTransformer
from config.model_config import ModelConfig
from config.experiment_config import ExperimentConfig
from data.cot_datasets import load_raw_dataset
from models.contemp_generator import ContemplationGenerator
from training.train_contemp_gen import train_contemplation_generator
from training.train_sent_trans import train_sentence_transformer, prepare_reasoning_pairs_dataset
from inference.inference import run_inference
from evaluation.metrics import evaluate_model
from baselines.baselines import run_baseline
from utils.logging import Logger
import utils.utils as utils

def parse_args():
    parser = argparse.ArgumentParser(description="Contemplation Tokens with Reasoning Ability")
    parser.add_argument("--num_exps", type=int, default=3, help="Number of experiments")
    parser.add_argument("--mode", type=str,
                        choices=["train_sentence_transformer", "train_contemp_generator",
                                 "evaluate", "baseline", "run_experiments", "train_ccot", "effi_cot"],
                        default="train_sentence_transformer",
                        help="Operation mode")
    parser.add_argument("--config", type=str, default="small",
                        choices=["small", "large", "mistral", "qwen"],
                        help="Configuration name. 'qwen' uses Qwen2.5-7B-Instruct as teacher and Qwen2.5-0.5B-Instruct as student")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "svamp", "multiarith", "commonsense_qa", "coin_flip"],
                        help="Dataset to use")
    parser.add_argument("--baseline", type=str, default="effi_cot",
                        choices=["cot", "ccot", "pause", "icot_kd", "nocot", "effi_cot", "icot_si", "codi", "softcot", "coconut"],
                        help="Baseline to run if mode is baseline")
    parser.add_argument("--ccot_stage", type=str, default="encode",choices=["encode", "decode", "prepare_decode_data", "evaluate", "cotrain_encode_decode"],
                        help="Stage for CCoT")
    # Add CODI stage for CODI baseline
    parser.add_argument("--codi_stage", type=str, default="train",
                                choices=["train", "evaluate"], help="Stage for CODI baseline")
    parser.add_argument("--coconut_stage", type=str, default="train",
                        choices=["train", "evaluate"], help="Stage for Coconut baseline")

    parser.add_argument("--experiment_file", type=str, default="experiments.json",
                        help="JSON file containing experiment configurations")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--variation", type=str, default="vanilla",
                        choices=["vanilla", "no_sentence_transformer", "no_l_reason", "no_warmup", "no_small_contemp_gen"],
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

    parser.add_argument("--st_linear_lr", "-stllr", type=float, default=1e-4,
                        help="Linear layer learning rate for sentence transformer")
    parser.add_argument("--st_linear_wd", "-stlwd", type=float, default=1e-3,
                        help="Linear layer weight decay for sentence transformer")
    parser.add_argument("--st_linear_epochs", "-stle", type=int, default=10,
                        help="Linear layer number of epochs for sentence transformer")
    parser.add_argument("--st_llm_lr","-stllmlr", type=float, default=1e-7,
                        help="LLM learning rate for sentence transformer")
    parser.add_argument("--st_llm_wd", "-stllmwd", type=float, default=1e-5,
                        help="LLM weight decay for sentence transformer")
    parser.add_argument("--st_llm_epochs", "-stllme", type=int, default=5,
                        help="LLM number of epochs for sentence transformer")
    parser.add_argument("--cg_linear_lr", "-cgllr", type=float, default=1e-4,
                        help="Linear layer learning rate for contemp generator")
    parser.add_argument("--cg_linear_wd", "-cglwd", type=float, default=1e-3,
                        help="Linear layer weight decay for contemp generator")
    parser.add_argument("--cg_linear_epochs", "-cgle", type=int, default=10,
                        help="Linear layer number of epochs for contemp generator")
    parser.add_argument("--cg_llm_lr","-cgllmlr", type=float, default=1e-7,
                        help="LLM learning rate  for contemp generator")
    parser.add_argument("--cg_llm_wd", "-cgllmwd", type=float, default=1e-5,
                        help="LLM weight decay  for contemp generator")
    parser.add_argument("--cg_llm_epochs", "-cgllme", type=int, default=5,
                        help="LLM number of epochs for contemp generator")
    # Add LoRA specific arguments
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="Rank for LoRA adapter")
    parser.add_argument("--lora_alpha", type=float, default=32,
                        help="Alpha scaling factor for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="Dropout probability for LoRA layers")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="alpha hyperparameter")
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    # Set random seed
    utils.set_seed(args.seed)

    # Print message if using Qwen configuration
    if args.config == "qwen":
        print("=" * 80)
        print("Using Qwen model configuration:")
        print("Teacher model: Qwen/Qwen2.5-7B-Instruct")
        print("Student model: Qwen/Qwen2.5-0.5B-Instruct")
        print("=" * 80)

    # Original logic for individual modes
    model_config = ModelConfig(args.config)
    experiment_config = ExperimentConfig(args.config)
    experiment_config.device = args.device
    experiment_config.ccot_stage = args.ccot_stage
    experiment_config.train_max_contemp_tokens = args.train_max_contemp_tokens if args.train_max_contemp_tokens is not None else experiment_config.train_max_contemp_tokens # SHOULD BE REMOVED, JUST DEBUG FOR DIFFERENT CONTEMP FOR CCOT
    experiment_config.eval_max_contemp_tokens = args.eval_max_contemp_tokens if args.eval_max_contemp_tokens is not None else experiment_config.eval_max_contemp_tokens # SHOULD BE REMOVED, JUST DEBUG FOR DIFFERENT CONTEMP FOR CCOT
    experiment_config.eval_temp = args.eval_temp

    # reset lr and wd and epochs of experiment config
    experiment_config.st_linear_lr = args.st_linear_lr
    experiment_config.st_linear_wd = args.st_linear_wd
    experiment_config.st_linear_epochs = args.st_linear_epochs
    experiment_config.st_llm_lr = args.st_llm_lr
    experiment_config.st_llm_wd = args.st_llm_wd
    experiment_config.st_llm_epochs = args.st_llm_epochs

    experiment_config.cg_linear_lr = args.cg_linear_lr
    experiment_config.cg_linear_wd = args.cg_linear_wd
    experiment_config.cg_linear_epochs = args.cg_linear_epochs
    experiment_config.cg_llm_lr = args.cg_llm_lr
    experiment_config.cg_llm_wd = args.cg_llm_wd
    experiment_config.cg_llm_epochs = args.cg_llm_epochs

    experiment_config.alpha = args.alpha

    if args.variation == "no_warmup":
        experiment_config.st_linear_epochs = 0
        experiment_config.cg_linear_epochs = 0

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
    elif args.mode == "baseline" and args.baseline == "coconut":
        experiment_config.model_save_path = f"{experiment_config.model_save_path}/coconut/{args.config}/{args.dataset}"
        experiment_config.checkpoint_path = f"{experiment_config.checkpoint_path}/coconut/{args.config}/{args.dataset}"
        experiment_config.result_path = f"{experiment_config.result_path}/coconut/{args.config}/{args.dataset}"
        experiment_config.experiment_name = f"coconut_{args.seed}_{args.dataset}_{args.config}"
        # Add Coconut stage to experiment config
        experiment_config.coconut_stage = args.coconut_stage
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

    reasoning_pairs_path = os.path.join(experiment_config.reasoning_pairs_path, f"{args.dataset}/reasoning_pairs_{args.seed}.json")

    # Load dataset
    if args.dataset == 'gsm8k':
        model_config.data_path = 'openai/gsm8k'
    elif args.dataset == 'svamp':
        model_config.data_path = 'ChilleD/SVAMP'
    elif args.dataset == 'multiarith':
        model_config.data_path = 'ChilleD/MultiArith'
    elif args.dataset == 'commonsense_qa':
        model_config.data_path = 'tau/commonsense_qa'
    elif args.dataset == 'coin_flip':
        model_config.data_path = 'skrishna/coin_flip'

    train_dataset, eval_dataset = load_raw_dataset(model_config.data_path)
    if args.mode == "effi_cot":
        logger = Logger(
            log_dir=experiment_config.log_dir,
            experiment_name=f"{experiment_config.experiment_name}"
        )
        logger.log_hyperparams(experiment_config.__dict__ | model_config.__dict__)
        # Prepare reasoning pairs dataset
        if os.path.exists(reasoning_pairs_path):
            pairs_dataset = utils.load_json(reasoning_pairs_path)
        else:
            # Extract queries from the dataset
            reasonings = None
            if "full_answer" in train_dataset[0].keys() and train_dataset[0]["full_answer"] != "":
                reasonings = [item["full_answer"] for item in train_dataset][:experiment_config.max_reasoning_pairs]
            pairs_dataset = prepare_reasoning_pairs_dataset(
                [item["query"] for item in train_dataset][:experiment_config.max_reasoning_pairs],
                reasonings,
                [item["answer"] for item in train_dataset][:experiment_config.max_reasoning_pairs],
                reasoning_pairs_path,
                max_pairs=experiment_config.max_reasoning_pairs
            )
    acc, ave_sample_time = [], []
    for i in range(args.num_exps):
        if args.mode == "effi_cot":
            cur_train_dataset, cur_val_dataset = load_raw_dataset(model_config.data_path)
            # add to train dataset items with condensed reasoning of pairs_dataset
            # for idx in range(len(pairs_dataset)):
            #     cur_train_dataset.update_item(idx, "condensed_reasoning", pairs_dataset[idx]["condensed_reasoning"])
            # sentence_transformer = None
            # if args.variation != "no_sentence_transformer":
            #     # Train sentence transformer
            #     sentence_transformer = train_sentence_transformer(
            #         model_config.teacher_model_name,
            #         experiment_config.start_layer_idx,
            #         experiment_config.end_layer_idx,
            #         cur_train_dataset,
            #         experiment_config
            #     )
            #     sentence_transformer = CustomizedSentenceTransformer.from_pretrained(
            #         experiment_config.model_save_path+"/sentence_transformer"
            #     ).to(args.device)

            # # Initialize contemplation generator
            # if args.variation == "no_small_contemp_gen":
            #     # For this variation, we use the teacher model with LoRA adapter
            #     contemp_generator = ContemplationGenerator(
            #         model_config.student_model_name,
            #         model_config.teacher_model_name,
            #         model_config.teacher_hidden_dim,
            #         device=args.device,
            #         variation="no_small_contemp_gen"
            #     )
            # else:
            #     # Use the standard approach with student model
            #     contemp_generator = ContemplationGenerator(
            #         model_config.student_model_name,
            #         model_config.teacher_model_name,
            #         model_config.teacher_hidden_dim,
            #         device=args.device,
            #         variation=args.variation
            #     )

            # # # Train the contemplation generator
            # train_contemplation_generator(
            #     contemp_generator,
            #     sentence_transformer,
            #     cur_train_dataset,
            #     cur_val_dataset,
            #     model_config,
            #     experiment_config,
            #     args.variation
            # )
            contemp_generator = ContemplationGenerator.from_pretrained(
                experiment_config.model_save_path+"/contemp_generator/"
            ).to(args.device)
            results = run_inference(
                contemp_generator,
                cur_val_dataset,
                model_config.teacher_model_name,
                experiment_config
            )
            # Evaluate results
            for temp, res in results:
                metrics = evaluate_model(res, eval_dataset)
                metrics.update({
                    'exp_num': i,
                    'dataset': args.dataset,
                    'eval_temp': temp,
                    'ave_sample_time': sum([r['sample_time'] for r in res]) / len(res),
                    'student': model_config.student_model_name,
                    'teacher': model_config.teacher_model_name,
                    'st_linear_lr': experiment_config.st_linear_lr,
                    'st_linear_wd': experiment_config.st_linear_wd,
                    'st_linear_epochs': experiment_config.st_linear_epochs,
                    'st_llm_lr': experiment_config.st_llm_lr,
                    'st_llm_wd': experiment_config.st_llm_wd,
                    'st_llm_epochs': experiment_config.st_llm_epochs,
                    'cg_linear_lr': experiment_config.cg_linear_lr,
                    'cg_linear_wd': experiment_config.cg_linear_wd,
                    'cg_linear_epochs': experiment_config.cg_linear_epochs,
                    'cg_llm_lr': experiment_config.cg_llm_lr,
                    'cg_llm_wd': experiment_config.cg_llm_wd,
                    'cg_llm_epochs': experiment_config.cg_llm_epochs,
                    'train_max_contemp_tokens': experiment_config.train_max_contemp_tokens,
                    'eval_max_contemp_tokens': experiment_config.eval_max_contemp_tokens,
                })
                utils.append_to_jsonl_file(f"{experiment_config.result_path}/evaluation_results.jsonl", metrics)
                print(f"Evaluation results: {metrics}")
            del contemp_generator, sentence_transformer
            torch.cuda.empty_cache()
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
            for temp, res in results:
                metrics = evaluate_model(res, eval_dataset)
                metrics.update({
                    'exp_num': i,
                    'dataset': args.dataset,
                    'eval_temp': temp,
                    'eval_max_contemp_tokens': experiment_config.eval_max_contemp_tokens,
                    'ave_sample_time': sum([r['sample_time'] for r in res]) / len(res)
                })
                utils.append_to_jsonl_file(f"{experiment_config.result_path}/evaluation_results.jsonl", metrics)
                print(f"Baseline {args.baseline} results: {metrics}")
        # acc.append(metrics["numerical_accuracy"])
        # ave_sample_time.append(sum([r['sample_time'] for r in results]) / len(results))
    # summary = {"summary_acc": f"{np.mean(acc):.2f} ± {np.std(acc):.2f}", "summary_time": f"{np.mean(ave_sample_time):.2f} ± {np.std(ave_sample_time):.2f}", "acc": acc, "ave_sample_time": ave_sample_time }
    # utils.append_to_jsonl_file(f"{experiment_config.result_path}/evaluation_results.jsonl", summary)
if __name__ == "__main__":
    main()