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
    codi_model_path = os.path.join(experiment_config.model_save_path, f"model_tokens={experiment_config.train_max_contemp_tokens}_lr={experiment_config.codi_lr}")
    
    print("Training model...")
    if not os.path.exists(codi_model_path):
        os.makedirs(codi_model_path, exist_ok=True)
        codi_model = train_codi_model(
            base_model_name=model_config.teacher_model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_path=codi_model_path,
            num_continuous_tokens=experiment_config.train_max_contemp_tokens,
            learning_rate=experiment_config.codi_lr,
            device=experiment_config.device
        )
    else:
        codi_model = CODIModel.from_pretrained(codi_model_path)
    codi_model.eval()
    
    print("Predicting on evaluation dataset...")
    results, gen_time = [], []
    with torch.no_grad():
        for sample in tqdm(eval_dataset, desc="Evaluating with CODI"):
            start_time = time.time()
            answers = codi_model.generate(sample, max_new_tokens=30 + experiment_config.eval_max_contemp_tokens,  temperature=experiment_config.eval_temp, top_p=0.9, do_sample=True)[-1]
            end_time = time.time()
            generation_time = end_time - start_time
            gen_time.append(generation_time)
            
            results.append({
                "query": sample['query'],
                "ground_truth": sample['answer'],
                "prediction": answers,
                "generation_time": generation_time
            })

    # Calculate total time
    ave_gen_time = sum(gen_time) / len(eval_dataset)

    # Save results
    results_dir = os.path.join(experiment_config.result_path, "codi")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Add summary statistics
    summary = {
        "avg_generation_time": ave_gen_time,
        "num_samples": len(eval_dataset),
        "num_continuous_tokens": experiment_config.eval_max_contemp_tokens
    }

    # Save results with summary
    utils.save_json({"results": results, "summary": summary}, f"{results_dir}/inference_results.json")
    print(f"CODI baseline completed. Average generation time: {ave_gen_time:.2f} seconds")
    os.remove(f"{codi_model_path}/model.pt")
    os.remove(f"{codi_model_path}/config.pt")
    os.removedirs(f"{codi_model_path}/checkpoints")
    return results