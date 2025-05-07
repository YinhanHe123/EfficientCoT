import shutil
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from training.train_pause import train_pause_model
from utils import utils

def run_pause_baseline(train_dataset, eval_dataset, model_config, experiment_config):
    """
    Implementation of the 'Think Before You Speak' baseline with pause tokens
    Based on: Goyal et al. "Think Before You Speak: Training Language Models With Pause Tokens"

    Args:
        train_dataset: Dataset for training (may be used for few-shot prompting)
        eval_dataset: Dataset for evaluation
        model_config: Model configuration
        experiment_config: Experiment configuration

    Returns:
        List of prediction results
    """
    device = "cuda:0"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    model = AutoModelForCausalLM.from_pretrained(model_config.teacher_model_name).to(device)

    # Create output directory for results
    result_dir = os.path.join(experiment_config.model_save_path, "pause")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Define pause token (we'll use a special token for this)
    pause_token = "<pause>"

    # Add pause token to tokenizer if it doesn't exist
    if pause_token not in tokenizer.get_vocab():
        special_tokens = {"additional_special_tokens": [pause_token]}
        num_added = tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))

    # Set default pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = train_pause_model(tokenizer, model, train_dataset, eval_dataset, experiment_config, result_dir)
    pause_token_id = tokenizer.convert_tokens_to_ids(pause_token)
    num_pause_tokens = experiment_config.eval_max_contemp_tokens
    pause_tokens = torch.tensor([[pause_token_id] * num_pause_tokens], device=device)
    all_res, all_summ = [], []
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results, gen_time = [], []
        with torch.no_grad():
            for sample in tqdm(eval_dataset, desc="Running Pause baseline"):
                query = sample["query"]

                # Format the prompt based on model type
                if "mistral" in model_config.teacher_model_name.lower():
                    prompt = f"<s>[INST] Question: {query}\nAnswer: [/INST]"
                else:
                    prompt = f"Question: {query}\nAnswer:"

                # Tokenize the input
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                start = time.time()
                # Append pause tokens to the input (the paper's core idea)
                modified_input_ids = torch.cat([input_ids, pause_tokens], dim=1)
                
                # Generate the response, ignoring the pause tokens
                outputs = model.generate(
                    modified_input_ids,
                    max_length=30 + modified_input_ids.size(1),
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True
                )
                end = time.time()

                # Skip input prompt and pause tokens to get just the answer
                answer_ids = outputs[0][input_ids.size(1) + num_pause_tokens:]
                answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                results.append({
                    "query": query,
                    "ground_truth": sample.get("answer", ""),
                    "prediction": answer,
                    "sample_time": end - start
                })
                gen_time.append(end - start)
        summary = {
            "avg_generation_time": sum(gen_time) / len(eval_dataset),
            "num_samples": len(eval_dataset),
            "num_continuous_tokens": experiment_config.eval_max_contemp_tokens
        }
        all_summ.append(summary)
        all_res.append((temp, results))
    
    results_dir = os.path.join(experiment_config.result_path, "pause")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.save_json([{"summary": summary} for summary in all_summ], f"{results_dir}/inference_results.json")
    return all_res