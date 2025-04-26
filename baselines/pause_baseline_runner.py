import time
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
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
    device = experiment_config.device

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    model = AutoModelForCausalLM.from_pretrained(model_config.teacher_model_name)
    model = model.to(device)
    model.eval()

    # Create output directory for results
    result_dir = os.path.join(experiment_config.result_path, "pause")
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

    results = []
    generation_times = []
    pause_token_id = tokenizer.convert_tokens_to_ids(pause_token)

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

            # Append pause tokens to the input (the paper's core idea)
            num_pause_tokens = experiment_config.eval_max_contemp_tokens

            # Create pause tokens tensor and append to input
            pause_tokens = torch.tensor([[pause_token_id] * num_pause_tokens], device=device)
            modified_input_ids = torch.cat([input_ids, pause_tokens], dim=1)
            
            start = time.time()
            # Generate the response, ignoring the pause tokens
            outputs = model.generate(
                modified_input_ids,
                max_length=150 + modified_input_ids.size(1),
                temperature=experiment_config.eval_temp,
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
                "generation_time": end - start
            })

    # Save results to file
    utils.save_json(results, f"{result_dir}/inference_results.json")

    return results