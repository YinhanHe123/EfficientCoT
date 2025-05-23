import torch
from tqdm import tqdm
import os
import time
from models.codi_model import CODIModel
from training.train_codi import train_codi_model
import utils.utils as utils

def run_codi_baseline(train_dataset, eval_dataset, model_config, experiment_config):
    """
    Continuous Chain-of-Thought via Self-Distillation (CODI) baseline
    Based on the paper "CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation"
    by Zhenyi Shen et al.

    Args:
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        model_config: Model configuration
        experiment_config: Experiment configuration

    Returns:
        List of prediction results
    """
    # Check for trained models
    output_path = f"{experiment_config.model_save_path}/model_tokens={experiment_config.train_max_contemp_tokens}_lr={experiment_config.codi_lr}"

    print("Training model...")
    if not os.path.exists(f"{output_path}/model.pt"):
        os.makedirs(output_path, exist_ok=True)
        train_codi_model(
            base_model_name=model_config.teacher_model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_path=output_path,
            num_continuous_tokens=experiment_config.train_max_contemp_tokens,
            learning_rate=experiment_config.codi_lr,
            device=experiment_config.device
        )
    codi_model = CODIModel.from_pretrained(output_path, experiment_config.device)
    codi_model.eval()

    print("Predicting on evaluation dataset...")
    all_res, all_summ = [], []
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results, gen_time = [], []
        with torch.no_grad():
            for sample in tqdm(eval_dataset, desc="Evaluating with CODI"):
                start_time = time.time()
                answers = codi_model.generate(sample, max_new_tokens=30 + experiment_config.eval_max_contemp_tokens,  temperature=temp, top_p=0.9, do_sample=True)[-1]
                end_time = time.time()
                generation_time = end_time - start_time
                gen_time.append(generation_time)

                results.append({
                    "query": sample['query'],
                    "ground_truth": sample['answer'],
                    "prediction": answers,
                    "sample_time": generation_time
                })

        # Calculate total time
        ave_gen_time = sum(gen_time) / len(eval_dataset)

        # Add summary statistics
        summary = {
            "avg_generation_time": ave_gen_time,
            "num_samples": len(eval_dataset),
            "num_continuous_tokens": experiment_config.eval_max_contemp_tokens
        }
        all_summ.append(summary)
        all_res.append((temp, results))

        print(f"CODI baseline completed. Average generation time: {ave_gen_time:.2f} seconds")

    results_dir = os.path.join(experiment_config.result_path, "codi")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.save_json([{"summary": summary} for summary in all_summ], f"{results_dir}/inference_results.json")
    # os.remove(f"{output_path}/model.pt")
    # os.remove(f"{output_path}/config.pt")
    # os.removedirs(f"{output_path}/checkpoints")
    return all_res