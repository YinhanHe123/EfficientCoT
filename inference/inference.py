from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import utils
import torch
from tqdm import tqdm
import os

def run_inference(contemp_generator, dataset, teacher_model_name, config):
    device = utils.get_device()
    contemp_generator = contemp_generator.to(device)
    contemp_generator.eval()

    # Load teacher LLM for generating answers
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # Create output directory for results
    result_dir = f"{config.result_path}/{config.experiment_name}"
    utils.create_directory(result_dir)

    results = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Running inference"):
            query = sample["query"]

            # Generate contemplation tokens hidden states (now acting as input embeddings)
            query_inputs = contemp_generator.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_seq_length // 2  # Leave room for the answer
            ).to(device)

            contemp_states = contemp_generator(
                query_inputs.input_ids,
                attention_mask=query_inputs.attention_mask
            )

            # Prepare prompt with query
            prompt = f"Question: {query}\nAnswer:"

            # Tokenize the prompt
            prompt_tokens = teacher_tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=config.max_seq_length // 2  # Leave room for the answer
            ).to(device)

            # Define a special token to separate prompt from contemplation
            # This helps the model distinguish between them
            separator_id = teacher_tokenizer.eos_token_id

            # Create input by concatenating prompt tokens with a separator
            input_ids = torch.cat([
                prompt_tokens.input_ids,
                torch.tensor([[separator_id]], device=device)
            ], dim=1)

            # Calculate the lengths for proper positioning
            prompt_length = input_ids.size(1)
            contemp_len = min(contemp_states.size(1), config.max_contemp_tokens)
            total_seq_length = prompt_length + contemp_len

            # Instead of using a hook to inject the contemplation states,
            # we'll implement a more direct approach using model's prepare_inputs_for_generation
            original_prepare_inputs = teacher_model.prepare_inputs_for_generation

            # Keep track of first call to avoid infinite recursion
            first_call = True

            def modified_prepare_inputs(input_ids, **kwargs):
                nonlocal first_call

                if first_call:
                    first_call = False

                    # Get the embeddings from the model's embedding layer
                    inputs_embeds = teacher_model.get_input_embeddings()(input_ids)

                    # Create a new inputs_embeds by concatenating with contemp_states
                    combined_embeds = torch.cat([
                        inputs_embeds,
                        contemp_states[:, :contemp_len, :]
                    ], dim=1)

                    # Create a proper attention mask that covers both parts
                    attention_mask = torch.ones(
                        (1, total_seq_length),
                        dtype=torch.long,
                        device=device
                    )

                    # Create position ids that account for both parts
                    position_ids = torch.arange(
                        total_seq_length,
                        dtype=torch.long,
                        device=device
                    ).unsqueeze(0)

                    # Return the combined inputs with proper positioning
                    kwargs.pop('attention_mask', None)  # Remove existing attention_mask
                    kwargs.pop('position_ids', None)   # Remove existing position_ids

                    return {
                        'inputs_embeds': combined_embeds,
                        'attention_mask': attention_mask,
                        'position_ids': position_ids,
                        **kwargs
                    }

                # For subsequent calls, use the original function
                return original_prepare_inputs(input_ids, **kwargs)

            # Replace the prepare_inputs_for_generation method temporarily
            teacher_model.prepare_inputs_for_generation = modified_prepare_inputs

            # Generate answer with the modified approach
            outputs = teacher_model.generate(
                input_ids,
                max_length=512 + input_ids.size(1),  # Account for the input length
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            # Decode only the generated part
            answer = teacher_tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)

            result = {
                "query": query,
                "ground_truth": sample.get("answer", ""),
                "prediction": answer
            }

            results.append(result)

    # if path not exist, create it
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Save results to file
    utils.save_json(results, f"{result_dir}/inference_results.json")

    return results