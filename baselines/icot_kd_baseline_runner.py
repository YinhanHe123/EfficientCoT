import torch
import os
from tqdm import tqdm
import utils.utils as utils
import gc
from training.train_icot_kd import train_icot_kd_model, ImplicitCoTModelWithRNN

def run_icot_kd_baseline(train_dataset, eval_dataset, model_config, experiment_config):
    """
    Implicit Chain of Thought baseline
    Based on: Deng et al. "Implicit Chain of Thought Reasoning via Knowledge Distillation"

    This implementation uses the RNN-based approach from the second project
    to implicitly model CoT reasoning.

    Args:
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        model_config: Model configuration
        experiment_config: Experiment configuration

    Returns:
        List of prediction results
    """
    # Check for trained model
    output_path = os.path.join(experiment_config.model_save_path, "icot_kd_model")

    # if model doesn't exist, train it
    if not os.path.exists(output_path):
        print("No pre-trained Implicit CoT model found. Training model...")
        os.makedirs(output_path, exist_ok=True)

        # Train the implicit CoT model
        train_icot_kd_model(
            teacher_model_name=model_config.teacher_model_name,
            student_model_name=model_config.student_model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_path=output_path,
            learning_rate=experiment_config.contemp_gen_lr,
            num_epochs=experiment_config.contemp_gen_epochs,
            batch_size=experiment_config.batch_size,
            device=experiment_config.device
        )
    icot_kd_model = ImplicitCoTModelWithRNN.from_pretrained(output_path, device=experiment_config.device)
    icot_kd_model.eval()

    # Run inference on evaluation dataset
    print("Predicting on evaluation dataset...")
    results = []

    # Helper function for formatting prompts
    def format_prompt(query):
        if "mistral" in model_config.teacher_model_name.lower():
            return f"<s>[INST] Question: {query}\n Generate the answer directly. Answer: [/INST]"
        elif "qwen" in model_config.teacher_model_name.lower():
            return f"<|im_start|>system\nYou are an expert in math word problems.<|im_end|>\n<|im_start|>user\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\nAnswer:"
        else:
            return f"Question: {query}\n Generate the answer directly. Answer:"

    with torch.no_grad():
        for sample in tqdm(eval_dataset, desc="Running Implicit CoT inference"):
            query = sample["query"]

            # Format prompt
            prompt = format_prompt(query)

            # Tokenize input
            inputs = icot_kd_model.student_tokenizer(
                f"Question: {sample['query']}\nAnswer: ",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=experiment_config.max_seq_length
            ).to(experiment_config.device)

            # Generate answer
            # First run through the implicit CoT model to get modified hidden states
            outputs = icot_kd_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )

            # Then use the student model for generation
            generated_outputs = icot_kd_model.student_model.generate(
                inputs.input_ids,
                max_new_tokens=30,
                temperature=experiment_config.eval_temp,
                top_p=0.9,
                do_sample=True,
                pad_token_id=icot_kd_model.student_tokenizer.eos_token_id
            )

            # Decode the output
            response = icot_kd_model.student_tokenizer.decode(
                generated_outputs[0][inputs.input_ids.size(1):],
                skip_special_tokens=True
            )

            answer = response.strip()

            # Save result
            results.append({
                "query": query,
                "ground_truth": sample.get("answer", ""),
                "prediction": answer
            })

    # Save results
    results_dir = os.path.join(experiment_config.result_path, "icot_kd")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    utils.save_json(results, f"{results_dir}/inference_results.json")

    # Clean up to save memory
    del icot_kd_model
    gc.collect()
    torch.cuda.empty_cache()

    return results