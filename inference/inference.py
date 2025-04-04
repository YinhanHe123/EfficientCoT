from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import utils
import torch
from tqdm import tqdm
import os
import time

def run_inference(contemp_generator, dataset, teacher_model_name, config):
    device = config.device
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
    time_list = []
    contemp_time_list = []
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Running inference"):
            config.max_contemp_tokens = 0
            query = sample["query"]

            # Generate contemplation tokens hidden states (now acting as input embeddings)
            query_inputs = contemp_generator.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_seq_length   # Leave room for the answer
            ).to(device)
            contemp_start = time.time()
            contemp_states = contemp_generator(
                query_inputs.input_ids,
                attention_mask=query_inputs.attention_mask
            )
            contemp_end = time.time()
            contemp_time = contemp_end - contemp_start
            contemp_time_list.append(contemp_time)

            # Prepare prompt with query
            prompt = f"Question: {query}\n Generate the answer directly. Answer:"
            # prompt = f"Question: {query}\n Think step by step. Answer:"

            # for debugging
            # prompt = [
            #     {'role':"user", "content": prompt}
            # ]
            # teacher_tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{ '\n\nHuman: ' + message['content'] +  eos_token }}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\n\nAssistant: ' }}{% endif %}"
            # prompt=teacher_tokenizer.apply_chat_template(prompt, tokenize=False)

            # Tokenize the prompt
            prompt_tokens = teacher_tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=config.max_seq_length  # Leave room for the answer
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

            # Remove the first_call mechanism and modify the prepare_inputs function
            def modified_prepare_inputs(input_ids, past_key_values=None, **kwargs):
                # If this is the first call (no past_key_values)
                if len(past_key_values.key_cache) == 0:
                    # Get the embeddings from the model's embedding layer
                    inputs_embeds = teacher_model.get_input_embeddings()(input_ids)

                    # Create a new inputs_embeds by concatenating with contemp_states
                    combined_embeds = torch.cat([
                        inputs_embeds,
                        contemp_states[:, -contemp_len:, :]
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
                    kwargs.pop('attention_mask', None)
                    kwargs.pop('position_ids', None)

                    return {
                        'inputs_embeds': combined_embeds,
                        'attention_mask': attention_mask,
                        'position_ids': position_ids,
                        **kwargs
                    }

                # For subsequent calls, adjust the past_key_values to include the effect of contemp_states
                else:
                    # The model will continue generating based on the KV cache that already includes
                    # the effect of the contemplation states from the first call
                    return original_prepare_inputs(input_ids, past_key_values=past_key_values, **kwargs)


            # Replace the prepare_inputs_for_generation method temporarily
            # teacher_model.prepare_inputs_for_generation = modified_prepare_inputs

            # Generate answer with the modified approach
            gen_start = time.time()
            outputs = teacher_model.generate(
                input_ids,
                # max_length=120 + input_ids.size(1),  # Account for the input length
                max_length = 30+input_ids.size(1)+contemp_len,
                temperature=config.eval_temp,
                top_p=0.9,
                do_sample=True
            )
            gen_end = time.time()
            gen_time = gen_end - gen_start
            # print('Time taken for teacher generation:', gen_time)
            # Decode only the generated part
            answer = teacher_tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)
            # print(f"Query: {query}\nAnswer: {answer}\n")
            # print('Answer', answer)

            result = {
                "query": query,
                "ground_truth": sample.get("answer", ""),
                "prediction": answer
            }
            time_list.append(contemp_time+gen_time)
            results.append(result)
            teacher_model.prepare_inputs_for_generation = original_prepare_inputs # change it back to original for the next sample in the loop

    # print(f"Average time taken for each sample: {sum(time_list)/len(time_list)}, Average time taken for contemplation: {sum(contemp_time_list)/len(contemp_time_list)}")
    # if path not exist, create it
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Save results to file
    utils.save_json(results, f"{result_dir}/inference_results.json")

    return results