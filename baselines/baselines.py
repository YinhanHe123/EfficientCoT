import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import os
from utils import utils

def run_baseline(baseline_name, dataset, model_config, experiment_config, num_shots=0):
    """Run one of the baseline methods for comparison"""

    if baseline_name == "ccot":
        return run_ccot_baseline(dataset, model_config, experiment_config)
    elif baseline_name == "pause":
        return run_pause_baseline(dataset, model_config, experiment_config)
    elif baseline_name == "implicit_cot":
        return run_implicit_cot_baseline(dataset, model_config, experiment_config)
    elif baseline_name == "cot":
        return run_cot_baseline(dataset, model_config, experiment_config, num_shots)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

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
            max_length=512,
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

def run_ccot_baseline(dataset, model_config, experiment_config):
    """
    Compressed Chain of Thought baseline
    Based on: Cheng & Van Durme "Compressed chain of thought: Efficient reasoning through dense representations"
    """
    import os
    from models.ccot_model import CCoTModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if we have a pre-trained CCoT model
    ccot_model_path = os.path.join(experiment_config.model_save_path, "ccot_model")
    if os.path.exists(ccot_model_path):
        # Load pre-trained CCoT model
        ccot_model = CCoTModel.from_pretrained(ccot_model_path)
        ccot_model = ccot_model.to(device)
        ccot_model.eval()
    else:
        # Initialize CCoT model
        ccot_model = CCoTModel(
            model_config.teacher_model_name,
            compression_ratio=0.1,  # Using a default 10x compression
            autoregressive_layer=15,  # Middle layer for autoregressive generation
            device=device
        )
        ccot_model = ccot_model.to(device)

    # Load teacher model for generation
    teacher_model = AutoModelForCausalLM.from_pretrained(model_config.teacher_model_name)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    results = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Running CCOT baseline"):
            query = sample["query"]

            # Prepare input for CCoT model
            input_text = f"Question: {query}\nAnswer:"
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=experiment_config.max_seq_length // 2  # Leave room for answer
            ).to(device)

            # Generate contemplation tokens
            contemplation_states = ccot_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask
            )

            # Prepare for teacher model generation
            # Get the input embeddings
            inputs_embeds = teacher_model.get_input_embeddings()(inputs.input_ids)

            # Create the combined inputs with contemplation states
            # Limit contemplation states to a reasonable length
            max_contemp_tokens = min(contemplation_states.size(1), 50)
            combined_embeds = torch.cat([
                inputs_embeds,
                contemplation_states[:, :max_contemp_tokens, :]
            ], dim=1)

            # Create proper attention mask for the combined sequence
            combined_attention_mask = torch.ones(
                (inputs.input_ids.size(0), combined_embeds.size(1)),
                dtype=torch.long,
                device=device
            )

            # Generate answer conditioned on query and contemplation tokens
            outputs = teacher_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                max_length=512 + combined_embeds.size(1),  # Account for input length
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            # Decode the generated answer
            # We need to skip the input length since we provided embeddings
            response = tokenizer.decode(outputs[0][inputs.input_ids.size(1):], skip_special_tokens=True)

            results.append({
                "query": query,
                "ground_truth": sample.get("answer", ""),
                "prediction": response
            })

    # Save results
    results_dir = experiment_config.result_path+"/ccot"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save results to file
    utils.save_json(results, f"{results_dir}/inference_results.json")

    return results

def run_pause_baseline(dataset, model_config, experiment_config):
    """
    Pause Tokens baseline
    Based on: Goyal et al. "Think before you speak: Training Language Models With Pause Tokens"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    model = AutoModelForCausalLM.from_pretrained(model_config.teacher_model_name)
    model = model.to(device)
    model.eval()

    # In the Pause method, the model has been trained with special pause tokens
    # Here we simulate this with a placeholder implementation

    results = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Running Pause baseline"):
            query = sample["query"]

            # Add pause token to prompt
            prompt = f"Question: {query}\n<PAUSE>\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                inputs.input_ids,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.replace(prompt, "").strip()

            results.append({
                "query": query,
                "ground_truth": sample.get("answer", ""),
                "prediction": answer
            })
    # save results
    return results

def run_implicit_cot_baseline(dataset, model_config, experiment_config):
    """
    Implicit Chain of Thought baseline
    Based on: Deng et al. "Implicit Chain of Thought Reasoning via Knowledge Distillation"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # In a real implementation, we would load a model that has been
    # distilled from a teacher using the Implicit CoT approach
    # For this placeholder, we'll use the teacher model directly

    tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    model = AutoModelForCausalLM.from_pretrained(model_config.teacher_model_name)
    model = model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Running Implicit CoT baseline"):
            query = sample["query"]

            # Standard prompt without explicit reasoning request
            prompt = f"Question: {query}\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                inputs.input_ids,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.replace(prompt, "").strip()

            results.append({
                "query": query,
                "ground_truth": sample.get("answer", ""),
                "prediction": answer
            })

    return results