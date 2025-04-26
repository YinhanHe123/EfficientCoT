import torch
from tqdm import tqdm
import os
import time
import gc
from models.softcot_model import SoftCoTModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import utils.utils as utils

def run_softcot_baseline(train_dataset, eval_dataset, model_config, experiment_config):
    """
    Run the SoftCoT baseline method

    Args:
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        model_config: Model configuration
        experiment_config: Experiment configuration

    Returns:
        List of prediction results
    """
    device = experiment_config.device

    # Paths for model components
    softcot_model_path = os.path.join(experiment_config.model_save_path, "softcot_model")

    # Training or loading the SoftCoT model
    if not os.path.exists(os.path.join(softcot_model_path, "config.pt")):
        print("No pre-trained SoftCoT model found. Training model...")
        os.makedirs(softcot_model_path, exist_ok=True)

        # Import the training function
        from training.train_softcot import train_softcot_model

        # Train the SoftCoT model
        softcot_model = train_softcot_model(
            llm_model_name=model_config.teacher_model_name,
            assistant_model_name=model_config.student_model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_path=softcot_model_path,
            learning_rate=experiment_config.contemp_gen_lr,
            weight_decay=experiment_config.contemp_gen_weight_decay,
            num_epochs=experiment_config.contemp_gen_epochs,
            batch_size=experiment_config.batch_size,
            num_soft_tokens=experiment_config.train_max_contemp_tokens,
            device=device
        )
    else:
        print("Loading pre-trained SoftCoT model...")
        softcot_model = SoftCoTModel.from_pretrained(softcot_model_path, device=device)

    # Set to evaluation mode
    softcot_model.assistant_model.eval()
    if softcot_model.projection_module:
        softcot_model.projection_module.eval()

    # Load LLM for inference
    print("Loading LLM for inference...")
    llm_model = AutoModelForCausalLM.from_pretrained(model_config.teacher_model_name)
    llm_model = llm_model.to(device)
    llm_model.eval()

    # Create output directory for results
    results_dir = os.path.join(experiment_config.result_path, "softcot")
    os.makedirs(results_dir, exist_ok=True)

    results = []
    softcot_time_list = []
    inference_time_list = []

    print("Running inference...")
    with torch.no_grad():
        for sample in tqdm(eval_dataset, desc="Running SoftCoT inference"):
            query = sample["query"]

            # Format the prompt based on the model type
            if "mistral" in model_config.teacher_model_name.lower():
                prompt = f"<s>[INST] Question: {query}\n Answer: [/INST]"
            else:
                prompt = f"Question: {query}\n Answer:"

            # Tokenize the prompt
            prompt_tokens = softcot_model.llm_tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=experiment_config.max_seq_length
            ).to(device)

            # Generate soft thought tokens
            start_time = time.time()
            num_soft_tokens = experiment_config.eval_max_contemp_tokens
            soft_thought_tokens = softcot_model.generate_soft_thoughts(query, num_soft_tokens)

            # Project to LLM space
            projected_soft_thoughts = softcot_model.project_soft_thoughts(soft_thought_tokens)
            softcot_time = time.time() - start_time
            softcot_time_list.append(softcot_time)

            # Get the embeddings from LLM's embedding layer
            inputs_embeds = llm_model.get_input_embeddings()(prompt_tokens.input_ids)

            # Concatenate with projected soft thoughts
            total_seq_length = prompt_tokens.input_ids.size(1) + projected_soft_thoughts.size(1)
            combined_embeds = torch.cat([
                inputs_embeds,
                projected_soft_thoughts
            ], dim=1)

            # Create proper attention mask
            attention_mask = torch.ones(
                (1, total_seq_length),
                dtype=torch.long,
                device=device
            )

            # Create position ids
            position_ids = torch.arange(
                total_seq_length,
                dtype=torch.long,
                device=device
            ).unsqueeze(0)

            # Generate answer
            start_time = time.time()
            outputs = llm_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=50 + total_seq_length,
                temperature=experiment_config.eval_temp,
                top_p=0.9,
                do_sample=True,
                pad_token_id=softcot_model.llm_tokenizer.eos_token_id
            )
            inference_time = time.time() - start_time
            inference_time_list.append(inference_time)

            # Extract the answer (skip the initial prompt and projected soft thoughts)
            answer_ids = outputs[0][prompt_tokens.input_ids.size(1):]
            answer = softcot_model.llm_tokenizer.decode(answer_ids, skip_special_tokens=True)

            # Record the result
            result = {
                "query": query,
                "ground_truth": sample.get("answer", ""),
                "prediction": answer,
                "softcot_time": softcot_time,
                "inference_time": inference_time,
                "total_time": softcot_time + inference_time
            }
            results.append(result)

    # Calculate average times
    avg_softcot_time = sum(softcot_time_list) / len(softcot_time_list)
    avg_inference_time = sum(inference_time_list) / len(inference_time_list)
    avg_total_time = avg_softcot_time + avg_inference_time

    # Add summary statistics
    summary = {
        "avg_softcot_time": avg_softcot_time,
        "avg_inference_time": avg_inference_time,
        "avg_total_time": avg_total_time,
        "num_samples": len(eval_dataset),
        "num_soft_tokens": experiment_config.eval_max_contemp_tokens
    }

    # Save results with summary
    utils.save_json({"results": results, "summary": summary},
                    f"{results_dir}/inference_results.json")

    print(f"SoftCoT baseline completed. Average generation time: {avg_total_time:.2f} seconds")

    # Clean up
    del llm_model
    gc.collect()
    torch.cuda.empty_cache()

    return results