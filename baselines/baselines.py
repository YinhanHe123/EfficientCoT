import time
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import os
from utils import utils
from .ccot_baseline_runner import run_ccot_baseline
from .icot_si_baseline_runner import run_icot_si_baseline
from .codi_baseline_runner import run_codi_baseline
from .icot_kd_baseline_runner import run_icot_kd_baseline
from .softcot_baseline_runner import run_softcot_baseline
from .pause_baseline_runner import run_pause_baseline
from .coconut_baseline_runner import run_coconut_baseline

def run_baseline(baseline_name, train_dataset, eval_dataset, model_config, experiment_config, num_shots=0):
    """Run one of the baseline methods for comparison"""

    if baseline_name == "ccot":
        return run_ccot_baseline(train_dataset, eval_dataset, model_config, experiment_config)
    elif baseline_name == "softcot":
        return run_softcot_baseline(train_dataset, eval_dataset, model_config, experiment_config)
    elif baseline_name == "codi":
        return run_codi_baseline(train_dataset, eval_dataset, model_config, experiment_config)
    elif baseline_name == "icot_si":
        return run_icot_si_baseline(train_dataset, eval_dataset, model_config, experiment_config)
    elif baseline_name == "pause":
        return run_pause_baseline(train_dataset, eval_dataset, model_config, experiment_config)
    elif baseline_name == "icot_kd":
        return run_icot_kd_baseline(train_dataset, eval_dataset, model_config, experiment_config)
    elif baseline_name == "coconut":
        return run_coconut_baseline(train_dataset, eval_dataset, model_config, experiment_config)
    elif baseline_name == "cot":
        return run_cot_baseline(train_dataset, eval_dataset, model_config, experiment_config, num_shots)
    elif baseline_name == "nocot":
        return run_nocot_baseline(eval_dataset, model_config, experiment_config, num_shots)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

def get_formatted_prompt(query):
    """
    Format the prompt based on the teacher model used
    """
    combined_input_for_query = f"[INST] Question: {query}"
    combined_input_for_answer = "Answer: "
    return (combined_input_for_query, combined_input_for_answer)

def run_cot_baseline(dataset, model_config, experiment_config, num_shots):
    # results is in shape of  = {"query": query, "ground_truth": sample.get("answer", ""),"prediction": answer}
    results = []
    pipe = pipeline("text-generation", model=model_config.teacher_model_name, tokenizer=model_config.teacher_model_name, device_map='auto')
    dem_samples = [
        # Headers and instructions
        "# 5-Shot Prompt for Math Word Problems",
        "Below are five examples of math word problems along with their step-by-step solutions. Use these examples to guide your approach when solving similar problems.",

        # Example 1
        "## Example 1:",
        "**Problem**: Emma purchased 3 notebooks for $4.50 each and 2 pens for $2.25 each. If she paid with a $20 bill, how much change did she receive?",
        "**Solution**:",
        "First, I'll calculate the total cost of the notebooks.",
        "3 notebooks × $4.50 = $13.50",
        "Next, I'll calculate the total cost of the pens.",
        "2 pens × $2.25 = $4.50",
        "Now, I'll find the total cost of her purchase.",
        "$13.50 + $4.50 = $18.00",
        "Finally, I'll calculate her change.",
        "$20.00 - $18.00 = $2.00",
        "Therefore, Emma received $2.00 in change.",

        # Example 2
        "## Example 2:",
        "**Problem**: A train travels at a speed of 80 kilometers per hour. If it departs at 9:15 AM and arrives at its destination at 12:45 PM, what is the distance between the two stations?",
        "**Solution**:",
        "First, I need to find the total travel time.",
        "Departure time: 9:15 AM",
        "Arrival time: 12:45 PM",
        "Time difference: 3 hours and 30 minutes = 3.5 hours",
        "Now I can calculate the distance using the formula: distance = speed × time",
        "Distance = 80 km/h × 3.5 h = 280 km",
        "Therefore, the distance between the two stations is 280 kilometers.",

        # Example 3
        "## Example 3:",
        "**Problem**: Carlos is making fruit salad for a party. He needs 2.5 cups of fruit per person. If 12 people will attend the party, and Carlos already has 8 cups of fruit, how many more cups of fruit does he need to prepare?",
        "**Solution**:",
        "First, I'll calculate the total amount of fruit needed for all 12 people.",
        "12 people × 2.5 cups per person = 30 cups",
        "Since Carlos already has 8 cups of fruit, I'll subtract to find out how many more cups he needs.",
        "30 cups - 8 cups = 22 cups",
        "Therefore, Carlos needs 22 more cups of fruit to prepare for the party.",

        # Example 4
        "## Example 4:",
        "**Problem**: The temperature on Monday was 72°F. On Tuesday, the temperature dropped by 8°F. On Wednesday, the temperature was 5°F higher than on Tuesday. What was the average temperature over these three days?",
        "**Solution**:",
        "Let me find the temperature for each day.",
        "Monday: 72°F",
        "Tuesday: 72°F - 8°F = 64°F",
        "Wednesday: 64°F + 5°F = 69°F",
        "Now I'll calculate the average by adding all temperatures and dividing by 3.",
        "(72°F + 64°F + 69°F) ÷ 3 = 205°F ÷ 3 = 68.33°F",
        "Therefore, the average temperature over the three days was 68.33°F.",

        # Example 5
        "## Example 5:",
        "**Problem**: A rectangular swimming pool is 25 meters long and 10 meters wide. Alex swims 8 complete laps around the perimeter of the pool. What is the total distance Alex swam?",
        "**Solution**:",
        "First, I'll calculate the perimeter of the pool.",
        "Perimeter = 2 × (length + width)",
        "Perimeter = 2 × (25 m + 10 m)",
        "Perimeter = 2 × 35 m",
        "Perimeter = 70 m",
        "Now I'll find the total distance for 8 laps.",
        "Total distance = 8 laps × 70 m",
        "Total distance = 560 m",
        "Therefore, Alex swam a total distance of 560 meters."
    ]
    # concatenate with predefined number of shots
    samples = ""
    for i in range(num_shots):
        samples += dem_samples[i]

    # Process each sample individually to avoid memory issues
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        query = data["query"]
        prompt = f"Use these examples to guide your approach when solving similar problems." + samples + f"\n\n{query}\n\nAnswer:"

        # Use the pipeline with explicit truncation parameter
        response = pipe(
            prompt,
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            truncation=True  # Explicitly activate truncation
        )

        answer = response[0]["generated_text"].replace(prompt, "").strip()
        print(answer)
        results.append({
            "query": query,
            "ground_truth": data.get("answer", ""),
            "prediction": answer
        })

    # save results
    # if path not exist, create it
    results_dir = experiment_config.result_path+"/cot"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save results to file
    utils.save_json(results, f"{results_dir}/inference_results.json")
    return results

def run_nocot_baseline(dataset, model_config, experiment_config, num_shots):
    device = experiment_config.device

    # Load teacher LLM for generating answers
    teacher_tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained(model_config.teacher_model_name).to(device)
    teacher_model.eval()
    
    results_dir = experiment_config.result_path+"/nocot"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    all_res, all_summ = [], []
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results, time_list, contemp_time_list = [], [], []
        with torch.no_grad():
            for sample in tqdm(dataset, desc="Running inference"):            
                query_prompt, answer_prompt = get_formatted_prompt(sample["query"])
                query_inputs = teacher_tokenizer(
                    query_prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    max_length=experiment_config.max_seq_length
                ).to(device)                
                answer_inputs = teacher_tokenizer(
                    answer_prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=experiment_config.max_seq_length,
                    add_special_tokens=False
                ).to(device)
                
                prompt_embeds_query = teacher_model.get_input_embeddings()(query_inputs.input_ids)
                prompt_embeds_answer = teacher_model.get_input_embeddings()(answer_inputs.input_ids)

                # Create combined embeddings
                combined_embeds = torch.cat([
                    prompt_embeds_query,
                    prompt_embeds_answer
                ], dim=1)

                # Create proper attention mask that covers both parts
                attention_mask = torch.ones(
                    (1, combined_embeds.size(1)),
                    dtype=torch.long,
                    device=device
                )
                
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
                    "sample_time": gen_time
                }
                time_list.append(gen_time)
                results.append(result)
            
            print(f"Average time taken for each sample: {sum(time_list)/len(time_list)}")
            all_res.append((temp, results))
            summary = {
                "avg_total_time": sum(time_list)/len(time_list),
                "num_samples": len(dataset),
                "num_contemp_tokens": 0
            }
            all_summ.append(summary)
    utils.create_directory(f"{results_dir}/inference_results.json")
    utils.save_json([{"summary": summary} for summary in all_summ], f"{results_dir}/inference_results.json")
    return all_res
