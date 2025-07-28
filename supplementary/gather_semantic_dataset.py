import sys
sys.path.append("..")
import argparse
import re
import torch
from tqdm import tqdm
from inference.inference import get_formatted_prompt
from data.cot_datasets import load_raw_dataset
from models.contemp_generator import ContemplationGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM

temp_map = {"gsm8k":0.5, "svamp":0.1, "multiarith":0.9, "commonsense_qa":0.5, "coin_flip":0.7}

def check_answer(dataset, prediction, ground_truth):
    if dataset == "commonsense_qa":
        return (ground_truth == prediction or
            ground_truth + '.' == prediction or
            ground_truth in prediction.split() or
            ground_truth + '.' in prediction.split())
    elif dataset == "coin_flip":
        return (ground_truth == prediction or
            (ground_truth == 'yes' and ('yes' in prediction.lower() or 'still heads' in prediction.lower())) or
            (ground_truth == 'no' and ('no' in prediction.lower() or 'not heads' in prediction.lower())))
    else:
        try:
            pred_nums = re.findall(r'-?\d+\.?\d*', prediction)
            pred_nums = [float(num) for num in pred_nums]
            gt_num = float(ground_truth)
            return any(abs(pred - gt_num) < 1e-6 for pred in pred_nums)  # Allow for small floating point differences
        except (ValueError, TypeError):
            return False

parser = argparse.ArgumentParser(description="Contemplation Token Semantics Experiments")
parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "svamp", "multiarith", "commonsense_qa", "coin_flip", "logiqa"],
                        help="Dataset to use")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--eval_num_contemp_tokens", type=int, default=1)
parser.add_argument("--config_name", type=str, default="small", choices=["small", "mistral"],
                        help="Config to use")
args = parser.parse_args()

if args.dataset == 'gsm8k':
    data_path = 'openai/gsm8k'
elif args.dataset == 'svamp':
    data_path = 'ChilleD/SVAMP'
elif args.dataset == 'multiarith':
    data_path = 'ChilleD/MultiArith'
elif args.dataset == 'commonsense_qa':
    data_path = 'tau/commonsense_qa'
elif args.dataset == 'coin_flip':
    data_path = 'skrishna/coin_flip'

if args.config_name == "small":
    teacher_model_name = "meta-llama/Llama-2-7b-chat-hf"
elif args.config_name == 'mistral':
    teacher_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

model_save_path = f"/data/nee7ne/effi_cot/saved_models/effi_cot/vanilla/small/{args.dataset}"
contemp_generator = ContemplationGenerator.from_pretrained(model_save_path+"/contemp_generator/").to(args.device)
_, eval_dataset = load_raw_dataset(data_path)

teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
teacher_model = teacher_model.to(args.device)
teacher_model.eval()

contemp_tokens = []
for sample in tqdm(eval_dataset, desc="Running inference"):
    query_prompt, answer_prompt = get_formatted_prompt(sample["query"])
    # Find the position where we inserted the contemplation tokens
    query_inputs = contemp_generator.tokenizer(
        query_prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        max_length=512
    ).to(args.device)
    prefix_length = query_inputs.input_ids.size(1) - 1

    answer_inputs = contemp_generator.tokenizer(
        answer_prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
        add_special_tokens=False
    ).to(args.device)

    # CHANGED: Extract contemplation states from the correct position (no longer the last tokens)
    query_inputs = torch.cat([
        query_inputs['input_ids'],
        torch.tensor([[contemp_generator.tokenizer.eos_token_id * args.eval_num_contemp_tokens]]).to(args.device),
        answer_inputs['input_ids']
    ], dim=1)

    # Get contemplation states from the correct position
    contemp_states = contemp_generator(
        query_inputs,
        attention_mask=torch.ones_like(query_inputs)
    )[:, prefix_length:prefix_length+args.eval_num_contemp_tokens, :]

    query_inputs = teacher_tokenizer(
        query_prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        max_length=512
    ).to(args.device)
    answer_inputs = teacher_tokenizer(
        answer_prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
        add_special_tokens=False
    ).to(args.device)

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
        device=args.device
    )

    # Generate answer with the combined embeddings directly
    outputs = teacher_model.generate(
        inputs_embeds=combined_embeds,
        attention_mask=attention_mask,
        max_length=30 + combined_embeds.size(1),
        temperature=temp_map[args.dataset],
        top_p=0.9,
        do_sample=True,
        pad_token_id=teacher_tokenizer.eos_token_id,
    )

    # Decode only the generated part (skip the prompt and contemplation tokens)
    prefix_length = combined_embeds.size(1)-1 if len(outputs) > combined_embeds.size(1) else 0
    answer = teacher_tokenizer.decode(outputs[0][prefix_length:], skip_special_tokens=True)

    if check_answer(args.dataset, answer, sample.get("answer", "")):
        contemp_tokens.append({"orig":sample['query'], "orig_token": contemp_states})
torch.save(contemp_tokens, f"./{args.dataset}_contemp_tokens.pt")