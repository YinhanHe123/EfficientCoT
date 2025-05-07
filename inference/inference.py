from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import utils
import torch
from tqdm import tqdm
import os
import time

def get_formatted_prompt(query):
    """
    Format the prompt based on the teacher model used
    """
    combined_input_for_query = f"[INST] Question: {query}"
    combined_input_for_answer = "Answer: "
    return (combined_input_for_query, combined_input_for_answer)


def run_inference(contemp_generator, dataset, teacher_model_name, config):
    device = config.device
    contemp_generator = contemp_generator.to(device)
    contemp_generator.eval()

    # Load teacher LLM for generating answers
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # Create output directory for results
    result_dir = f"{config.result_path}/{config.experiment_name}"

    all_res, all_summ = [], []
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results, time_list, contemp_time_list = [], [], []
        with torch.no_grad():
            for sample in tqdm(dataset, desc="Running inference"):            
                query_prompt, answer_prompt = get_formatted_prompt(sample["query"])
                # Find the position where we inserted the contemplation tokens
                query_inputs = contemp_generator.tokenizer(
                    query_prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    max_length=config.max_seq_length
                ).to(device)
                prefix_length = query_inputs.input_ids.size(1) - 1
                
                answer_inputs = contemp_generator.tokenizer(
                    answer_prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=config.max_seq_length,
                    add_special_tokens=False
                ).to(device)
                
                # CHANGED: Extract contemplation states from the correct position (no longer the last tokens)
                query_inputs = torch.cat([
                    query_inputs['input_ids'], 
                    torch.tensor([[contemp_generator.tokenizer.eos_token_id * config.eval_max_contemp_tokens]]).to(device), 
                    answer_inputs['input_ids']
                ], dim=1)

                contemp_start = time.time()
                # Get contemplation states from the correct position
                contemp_states = contemp_generator(
                    query_inputs,
                    attention_mask=torch.ones_like(query_inputs)
                )[:, prefix_length:prefix_length+config.eval_max_contemp_tokens, :]

                contemp_end = time.time()
                contemp_time = contemp_end - contemp_start
                contemp_time_list.append(contemp_time)
                
                query_inputs = teacher_tokenizer(
                    query_prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    max_length=config.max_seq_length
                ).to(device)                
                answer_inputs = teacher_tokenizer(
                    answer_prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=config.max_seq_length,
                    add_special_tokens=False
                ).to(device)
            
                # Instead of modifying prepare_inputs_for_generation,
                # directly create input embeddings and concatenate with contemplation states
                prompt_embeds_query = teacher_model.get_input_embeddings()(query_inputs.input_ids)
                prompt_embeds_answer = teacher_model.get_input_embeddings()(answer_inputs.input_ids)

                # Create combined embeddings
                combined_embeds = torch.cat([
                    prompt_embeds_query,
                    contemp_states,
                    prompt_embeds_answer
                ], dim=1)

                # Create proper attention mask that covers both parts
                attention_mask = torch.ones(
                    (1, combined_embeds.size(1)),
                    dtype=torch.long,
                    device=device
                )

                # Generate answer with the combined embeddings directly
                gen_start = time.time()
                outputs = teacher_model.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=attention_mask,
                    max_length=30 + combined_embeds.size(1),
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=teacher_tokenizer.eos_token_id,
                )
                gen_end = time.time()
                # print(f"Generation time: {gen_end - gen_start}")
                gen_time = gen_end - gen_start

                # Decode only the generated part (skip the prompt and contemplation tokens)
                prefix_length = combined_embeds.size(1)-1 if len(outputs) > combined_embeds.size(1) else 0
                answer = teacher_tokenizer.decode(outputs[0][prefix_length:], skip_special_tokens=True)

                result = {
                    "query": sample["query"],
                    "ground_truth": sample.get("answer", ""),
                    "prediction": answer,
                    "sample_time": contemp_time + gen_time
                }
                time_list.append(contemp_time + gen_time)
                results.append(result)

                # Clean up memory
                del query_inputs, answer_inputs, prompt_embeds_query, prompt_embeds_answer, combined_embeds, contemp_states, outputs
                torch.cuda.empty_cache()
            
            print(f"Average time taken for each sample: {sum(time_list)/len(time_list)}, Average time taken for contemplation: {sum(contemp_time_list)/len(contemp_time_list)}")
            all_res.append((temp, results))
            summary = {
                "avg_total_time": sum(time_list)/len(time_list),
                "avg_comp_time": sum(contemp_time_list)/len(contemp_time_list),
                "num_samples": len(dataset),
                "num_contemp_tokens": config.eval_max_contemp_tokens
            }
            all_summ.append(summary)

    # Save results to file
    utils.create_directory(f"{result_dir}/inference_results.json")
    utils.save_json([{"summary": summary} for summary in all_summ], f"{result_dir}/inference_results.json")
    return all_res

# def run_inference(contemp_generator, dataset, teacher_model_name, config):
    # device = config.device
    # contemp_generator = contemp_generator.to(device)
    # contemp_generator.eval()

    # # Load teacher LLM for generating answers
    # teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    # teacher_tokenizer.pad_token = teacher_tokenizer.eos_token  # Set pad token to end of sequence token
    # teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    # teacher_model = teacher_model.to(device)
    # teacher_model.eval()

    # # Create output directory for results
    # result_dir = f"{config.result_path}/{config.experiment_name}"
    # utils.create_directory(result_dir)

    # results = []
    # time_list = []
    # contemp_time_list = []
    # with torch.no_grad():
    #     for sample in tqdm(dataset, desc="Running inference"):
    #         # config.eval_max_contemp_tokens = 0
    #         query = sample["query"]

    #         # Format the prompt based on the model
    #         if "mistral" in teacher_model_name.lower():
    #             query_condensed_reasoning = f"<s>[INST] You are an expert in math word problems. Question: {query}\nAnswer: [/INST]"
    #         else:
    #             query_condensed_reasoning = f"<<SYS>>You are an expert in math word problems<</SYS>>\nQuestion: {query}\nAnswer: "

    #         query_condensed_reasoning += f"{contemp_generator.tokenizer.eos_token} " * config.eval_max_contemp_tokens
    #         query_condensed_reasoning = query_condensed_reasoning.strip()

    #         # Generate contemplation tokens hidden states (now acting as input embeddings)
    #         query_inputs = contemp_generator.tokenizer(
    #             query_condensed_reasoning,
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=True,
    #             max_length=config.max_seq_length   # Leave room for the answer
    #         ).to(device)

    #         contemp_start = time.time()
    #         contemp_states = contemp_generator(
    #             query_inputs.input_ids,
    #             attention_mask=query_inputs.attention_mask
    #         )[:, -config.eval_max_contemp_tokens:, :]
    #         contemp_end = time.time()
    #         contemp_time = contemp_end - contemp_start
    #         contemp_time_list.append(contemp_time)

    #         # Prepare prompt with query based on model
    #         prompt = get_formatted_prompt(query, teacher_model_name)

    #         # Tokenize the prompt
    #         prompt_tokens = teacher_tokenizer(
    #             prompt,
    #             return_tensors="pt",
    #             padding=False,
    #             truncation=True,
    #             max_length=config.max_seq_length  # Leave room for the answer
    #         ).to(device)

    #         # Define a special token to separate prompt from contemplation
    #         # This helps the model distinguish between them
    #         separator_id = teacher_tokenizer.eos_token_id

    #         # Create input by concatenating prompt tokens with a separator
    #         input_ids = torch.cat([
    #             prompt_tokens.input_ids,
    #             torch.tensor([[separator_id]], device=device)
    #         ], dim=1)

    #         # Calculate the lengths for proper positioning
    #         prompt_length = input_ids.size(1)
    #         contemp_len = min(contemp_states.size(1), config.eval_max_contemp_tokens)
    #         total_seq_length = prompt_length + contemp_len

    #         # Instead of using a hook to inject the contemplation states,
    #         # we'll implement a more direct approach using model's prepare_inputs_for_generation
    #         original_prepare_inputs = teacher_model.prepare_inputs_for_generation

    #         # Remove the first_call mechanism and modify the prepare_inputs function
    #         def modified_prepare_inputs(input_ids, past_key_values=None, **kwargs):
    #             if past_key_values is None or len(past_key_values) == 0:
    #                 # Get the embeddings from the model's embedding layer
    #                 inputs_embeds = teacher_model.get_input_embeddings()(input_ids)

    #                 # Create a new inputs_embeds by concatenating with contemp_states
    #                 combined_embeds = torch.cat([
    #                     inputs_embeds,
    #                     contemp_states[:, -contemp_len:, :]
    #                 ], dim=1)

    #                 # Create a proper attention mask that covers both parts
    #                 attention_mask = torch.ones(
    #                     (1, total_seq_length),
    #                     dtype=torch.long,
    #                     device=device
    #                 )

    #                 # Create position ids that account for both parts
    #                 position_ids = torch.arange(
    #                     total_seq_length,
    #                     dtype=torch.long,
    #                     device=device
    #                 ).unsqueeze(0)

    #                 # Return the combined inputs with proper positioning
    #                 kwargs.pop('attention_mask', None)
    #                 kwargs.pop('position_ids', None)

    #                 return {
    #                     'inputs_embeds': combined_embeds,
    #                     'attention_mask': attention_mask,
    #                     'position_ids': position_ids,
    #                     **kwargs
    #                 }

    #             # For subsequent calls, adjust the past_key_values to include the effect of contemp_states
    #             else:
    #                 # The model will continue generating based on the KV cache that already includes
    #                 # the effect of the contemplation states from the first call
    #                 return original_prepare_inputs(input_ids, past_key_values=past_key_values, **kwargs)


    #         # Replace the prepare_inputs_for_generation method temporarily
    #         teacher_model.prepare_inputs_for_generation = modified_prepare_inputs

    #         # Generate answer with the modified approach
    #         gen_start = time.time()
    #         outputs = teacher_model.generate(
    #             input_ids,
    #             max_length = 30+input_ids.size(1)+contemp_len,
    #             temperature=config.eval_temp,
    #             top_p=0.9,
    #             do_sample=True,
    #             pad_token_id=teacher_tokenizer.eos_token_id,
    #         )
    #         gen_end = time.time()
    #         gen_time = gen_end - gen_start
    #         print(f"Generation time: {gen_time}")

    #         # Decode only the generated part
    #         answer = teacher_tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)
    #         # print(f"Answer: {answer}")

    #         result = {
    #             "query": query,
    #             "ground_truth": sample.get("answer", ""),
    #             "prediction": answer,
    #             "sample_time": contemp_time+gen_time
    #         }
    #         time_list.append(contemp_time+gen_time)
    #         results.append(result)
    #         teacher_model.prepare_inputs_for_generation = original_prepare_inputs # change it back to original for the next sample in the loop

    # print(f"Average time taken for each sample: {sum(time_list)/len(time_list)}, Average time taken for contemplation: {sum(contemp_time_list)/len(contemp_time_list)}")
    # # if path not exist, create it
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)

    # # Save results to file
    # utils.save_json(results, f"{result_dir}/inference_results.json")

    # return results