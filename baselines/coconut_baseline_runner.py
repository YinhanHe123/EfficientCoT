import torch
from tqdm import tqdm
import os
import time
from models.coconut_model import CoconutModel
import utils.utils as utils
import gc
import sys

def run_coconut_baseline(train_dataset, eval_dataset, model_config, experiment_config):
    """
    Chain of Continuous Thought (Coconut) baseline
    Based on: Hao et al. "Training Large Language Models to Reason in a Continuous Latent Space"

    Args:
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        model_config: Model configuration
        experiment_config: Experiment configuration

    Returns:
        List of prediction results
    """
    device = experiment_config.device

    # Check if trained Coconut model exists
    coconut_model_path = os.path.join(experiment_config.model_save_path, "coconut_model")

    # Stage handling: train or evaluate
    # if experiment_config.coconut_stage == "train":
    # Import training function
    from training.train_coconut import train_coconut_model

    # Train the Coconut model
    os.makedirs(coconut_model_path, exist_ok=True)

    coconut_model = train_coconut_model(
        base_model_name=model_config.teacher_model_name,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_path=coconut_model_path,
        learning_rate=experiment_config.ccot_lr,  # Reuse CCOT learning rate
        num_epochs=1,
        batch_size=1,  # Coconut can be memory intensive
        max_continuous_tokens=experiment_config.train_max_contemp_tokens,
        device=device
    )

    print("Coconut model training completed!")



    # elif experiment_config.coconut_stage == "evaluate":
        # Check if model exists
    if not os.path.exists(os.path.join(coconut_model_path, "model.pt")):
        print("No trained Coconut model found. Please train the model first (coconut_stage=train).")
        return None

    print("Evaluating with Coconut model...")
    all_res, all_summ = [], []
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results = []
        inference_times = []

        # Load the Coconut model
        coconut_model = CoconutModel.from_pretrained(coconut_model_path)
        coconut_model = coconut_model.to(device)
        coconut_model.eval()

        with torch.no_grad():
            for sample in tqdm(eval_dataset, desc="Running inference"):
                query = sample["query"]

                # Start timing
                start_time = time.time()

                # Generate answer with continuous thoughts
                answer = coconut_model.generate_with_continuous_thoughts(
                    query,
                    max_continuous_tokens=experiment_config.eval_max_contemp_tokens,
                    max_new_tokens=30 + experiment_config.eval_max_contemp_tokens,
                    temperature=temp,
                )

                # End timing
                end_time = time.time()
                inference_time = end_time - start_time
                inference_times.append(inference_time)

                # Store result
                results.append({
                    "query": query,
                    "ground_truth": sample.get("answer", ""),
                    "prediction": answer,
                    "sample_time": inference_time
                })

        # Free memory
        del coconut_model
        gc.collect()
        torch.cuda.empty_cache()

        # Calculate average inference time
        avg_time = sum(inference_times) / len(inference_times)

        # Add summary statistics
        summary = {
            "avg_generation_time": avg_time,
            "num_samples": len(eval_dataset),
            "max_continuous_tokens": experiment_config.eval_max_contemp_tokens
        }
        
        all_summ.append(summary)
        all_res.append((temp, results))

        print(f"Coconut baseline completed. Average generation time: {avg_time:.2f} seconds")
    # Save results
    results_dir = os.path.join(experiment_config.result_path, "coconut")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.save_json([{"summary": summary} for summary in all_summ], f"{results_dir}/inference_results.json")
    return all_res

  