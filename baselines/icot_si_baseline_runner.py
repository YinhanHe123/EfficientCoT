import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteriaList
from peft import get_peft_model, LoraConfig, TaskType
import utils.utils as utils

# Utilities from the second project adapted to the first project's framework
class DoubleEOSStoppingCriteria(StoppingCriteriaList):
    """Stop generation only after generating two EOSs, such as z <eos> y <eos>"""
    def __init__(self, eos_token_id):
        super().__init__([])
        self.eos_token_id_ = eos_token_id
        self.init = False

    def __call__(self, input_ids, scores):
        eos_count = (input_ids == self.eos_token_id_).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        return done.all()

class DoubleEOSLogitsProcessor(nn.Module):
    """Process logits to control EOS generation for CoT"""
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id_ = eos_token_id
        self.init = False

    def __call__(self, input_ids, scores):
        eos_count = (input_ids == self.eos_token_id_).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        if done.any():
            scores[done, :] = float('-inf')
            scores[done, self.eos_token_id_] = 0
        return scores

def get_sep_position(input_ids, sep_id, skip=0):
    """Get separator token positions in input_ids"""
    batch_size = input_ids.shape[0]
    sep_positions = input_ids.new_zeros(batch_size).long()
    for batch_id in range(batch_size):
        mask = input_ids[batch_id].eq(sep_id)
        if mask.sum() == 0:
            sep_positions[batch_id] = input_ids.shape[1] - 1
            continue
        sep_position = mask.nonzero()[skip] if mask.sum() > skip else mask.nonzero()[0]
        sep_positions[batch_id] = sep_position
    return sep_positions

def batch_ids(input_ids_list, pad_token_id, device, dtype):
    """Batch input ids with padding"""
    max_seq_len = max([len(item) for item in input_ids_list])
    batch_size = len(input_ids_list)
    input_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=dtype, device=device)
    for batch_id in range(batch_size):
        input_ids[batch_id, :len(input_ids_list[batch_id])] = input_ids_list[batch_id]
    return input_ids

def compute_lambda_distribution(removal_smoothing_lambda, truncate_length=100):
    """Compute lambda distribution for removal smoothing"""
    if removal_smoothing_lambda == float('inf'):
        lambda_distribution = torch.zeros(truncate_length)
        lambda_distribution[0] = 1
    else:
        positions = torch.arange(truncate_length)
        lambda_distribution = (1 - math.exp(-removal_smoothing_lambda)) * positions.mul(-removal_smoothing_lambda).exp()
        cum_prob = lambda_distribution.sum()
        assert cum_prob <= 1
        lambda_distribution[-1] = lambda_distribution[-1] + (1-cum_prob)
    return lambda_distribution

def extract_answer(text, eos_token):
    """Extract the final answer from text with eos_token delimiter"""
    possible_answers = text.split(eos_token)
    possible_answers = [a.strip() for a in possible_answers if len(a.strip()) > 0]
    return possible_answers[-1]

class ImplicitCoTTrainer:
    """Trainer for Implicit Chain of Thought (adapted from second project)"""
    def __init__(
        self,
        model_name,
        train_dataset,
        eval_dataset,
        output_path,
        device='cuda',
        epochs=10,
        lr=5e-5,
        batch_size=8,
        max_seq_length=512,
        remove_per_epoch=8,
        removal_smoothing_lambda=float('inf'),
        removal_side='left',
        keep_position=True,
        rank=128, 
        alpha=32
    ):
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_path = output_path
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.remove_per_epoch = remove_per_epoch
        self.removal_smoothing_lambda = removal_smoothing_lambda
        self.removal_side = removal_side
        self.keep_position = keep_position

        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.model.to(device)

        # Initialize removal parameters
        self.lambda_distribution = compute_lambda_distribution(removal_smoothing_lambda)
        
        # apply lora
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
        )

        # Enable input require grads
        self.model.enable_input_require_grads()

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)

        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def prepare_item(self, item, scheduled_to_remove=0, with_answer=True):
        """Prepare a batch for training with CoT removal strategy"""
        eos_tok = self.tokenizer.eos_token
        if "reasoning" in item and len(item["reasoning"]) > 0:
            steps = item["reasoning"].split("\n")
            cot = "\n".join(steps[:-1] if len(steps) > 1 else steps)
        elif "full_answer" in item and len(item["full_answer"]) > 0:
            steps = item["full_answer"].split("### Conclusion")
            cot = steps[0]
            if len(steps) == 1:
                steps = item["full_answer"].split("\n\n")
                cot = "\n\n".join(steps[:-1])
        input_text = f"{item['query']} {eos_tok} {cot} {eos_tok}"
        if with_answer:
            input_text += f" {item['answer']} {eos_tok}"

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.device)

        input_ids, labels = inputs.input_ids.clone(), inputs.input_ids.clone()

        # Find separator positions (question|reasoning|answer)
        question_end = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        reasoning_delimiter = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=1)

        # Mask out question tokens in labels (we don't want to predict the question)
        labels[0, :question_end[0]+1] = self.tokenizer.pad_token_id

        # Apply CoT removal if scheduled
        if scheduled_to_remove > 0:
            if (question_end[0] + 1 + scheduled_to_remove) > reasoning_delimiter[0]:
                return None
            
            position_ids = None
            if self.keep_position:
                position_ids = torch.arange(0, input_ids.shape[1], device=self.device).unsqueeze(0)

            # Sample random removal offsets
            random_removal_offset = torch.multinomial(
                self.lambda_distribution, 1, replacement=True
            ).to(self.device)

            to_remove = scheduled_to_remove + random_removal_offset

            if self.removal_side == 'left':
                # Remove from start of reasoning
                removal_from = question_end + 1
                removal_to = removal_from + to_remove
            else:
                # Remove from end of reasoning
                removal_to = reasoning_delimiter
                removal_from = reasoning_delimiter - to_remove

            qe = question_end[0] + 1
            rd = reasoning_delimiter[0] if reasoning_delimiter[0] > 0 else input_ids.shape[1] - 1

            # Calculate actual removal positions
            r_from, r_to = max(removal_from[0], qe), min(removal_to[0], rd)

            if self.keep_position and position_ids is not None:
                position_ids[0, r_from:] += r_to - r_from

            # Create new sequences with removed tokens
            input_ids = torch.cat([
                input_ids[0, :r_from], input_ids[0, r_to:]
            ]).unsqueeze(0)

            labels = torch.cat([
                labels[0, :r_from], labels[0, r_to:]
            ]).unsqueeze(0)
            
            # Batch padded sequences
            input_ids = batch_ids(input_ids, self.tokenizer.pad_token_id, self.device, input_ids.dtype)
            labels = batch_ids(labels, self.tokenizer.pad_token_id, self.device, labels.dtype)

            return {
                'input_ids': input_ids,
                'labels': labels,
                'position_ids': position_ids[:, :input_ids.shape[1]] if position_ids is not None else None
            }

        return {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': None
        }

    def train_epoch(self, scheduled_to_remove=0):
        """Train for one epoch with the specified amount of CoT removal"""
        self.model.train()
        total_loss, steps, max_len_reached = 0, 0, 0
        for item in tqdm(self.train_dataset, desc=f"Training (remove={scheduled_to_remove})"):
            data = self.prepare_item(item, scheduled_to_remove)
            if data is None:
                max_len_reached += 1
                continue
            # Forward pass
            outputs = self.model(
                input_ids=data['input_ids'],
                labels=data['labels'],
                position_ids=data['position_ids']
            )

            shift_logits =  outputs.logits[..., :-1, :].contiguous()
            shift_labels = data['labels'][..., 1:].contiguous()
            loss = torch.nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

            if steps % 100 == 0:
                print(f"Step {steps} - Loss: {loss.item():.4f}")
        if max_len_reached == len(self.train_dataset):
            return -1
        return total_loss / steps

    def evaluate(self, scheduled_to_remove=0):
        """Evaluate the model on the evaluation dataset"""
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for item in tqdm(self.eval_dataset, desc=f"Evaluating (remove={scheduled_to_remove})"):
                data = self.prepare_item(item, scheduled_to_remove, with_answer=False)
                if data is None:
                    continue
                # Generate answers
                stopping_criteria = DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)
                logits_processor = DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)
                generation_config = GenerationConfig.from_model_config(self.model.config)
                generation_config.eos_token_id = -1

                outputs = self.model.generate(
                    input_ids=data['input_ids'],
                    position_ids=data['position_ids'],
                    max_new_tokens=30,
                    stopping_criteria=[stopping_criteria],
                    logits_processor=[logits_processor],
                    generation_config=generation_config,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

                # Compare generated answers with ground truth
                generated_text = self.tokenizer.decode(outputs[0])
                generated_answer = extract_answer(generated_text, self.tokenizer.eos_token)
                if generated_answer.strip() == item["answer"].strip():
                    total_correct += 1
        return total_correct / len(self.eval_dataset)

    def train(self):
        """Train the model with progressive CoT removal"""
        best_accuracy = 0
        for epoch in range(self.epochs):
            scheduled_to_remove = epoch * self.remove_per_epoch
            print(f"Epoch {epoch+1}/{self.epochs} - Removing {scheduled_to_remove} tokens")

            # Train for one epoch
            avg_loss = self.train_epoch(scheduled_to_remove)
            if avg_loss == -1: break
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

            # Evaluate
            accuracy = self.evaluate(scheduled_to_remove)
            print(f"Epoch {epoch+1} - Accuracy: {accuracy:.4f}")

            # Save checkpoint
            self.model.save_pretrained(self.output_path)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.model.save_pretrained(self.output_path)
                print(f"New best accuracy: {best_accuracy:.4f}")
        print(f"Training completed. Best accuracy: {best_accuracy:.4f}")

def run_icot_si_baseline(train_dataset, eval_dataset, model_config, experiment_config):
    """
    Implicit Chain of Thought Stepwise Internalization baseline
    Based on: "From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step"

    Args:
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        model_config: Model configuration
        experiment_config: Experiment configuration

    Returns:
        List of prediction results
    """
    # Set up output paths
    output_path = f"{experiment_config.model_save_path}/model"

    # Check if the model has already been trained
    if not os.path.exists(f"{output_path}/adapter_model.safetensors"):
        os.makedirs(output_path, exist_ok=True)
        print("Training Implicit CoT model...")

        # Initialize trainer
        trainer = ImplicitCoTTrainer(
            model_name=model_config.teacher_model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_path=output_path,
            device=experiment_config.device,
            batch_size=1,
            max_seq_length=experiment_config.max_seq_length,
            remove_per_epoch=8,  # Gradually increase removal tokens
            removal_smoothing_lambda=4,  # Allow some variance in removal
            removal_side='left',  # Remove from start of reasoning
            keep_position=True,  # Maintain position embeddings
        )
        # Train the model
        trainer.train()
        del trainer.model, trainer
        torch.cuda.empty_cache()
    # Load the trained model for inference
    tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(output_path).to(experiment_config.device)
    model.eval()
    
    stopping_criteria = DoubleEOSStoppingCriteria(tokenizer.eos_token_id)
    logits_processor = DoubleEOSLogitsProcessor(tokenizer.eos_token_id)
    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.eos_token_id = -1

    # Prepare inference results
    results = []
    # Run inference on evaluation dataset
    for item in tqdm(eval_dataset, desc="Running Implicit CoT inference"):
        # Tokenize input
        input_text = f"{item['query']} {tokenizer.eos_token}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=experiment_config.max_seq_length
        ).to(experiment_config.device)

        # Generate answer
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=30,
                stopping_criteria=[stopping_criteria],
                logits_processor=[logits_processor],
                generation_config=generation_config,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        end = time.time()

        # Decode and extract answer
        generated_text = tokenizer.decode(outputs[0])
        answer = extract_answer(generated_text, tokenizer.eos_token)

        # Store result
        results.append({
            "query": item['query'],
            "ground_truth": item.get("answer", ""),
            "prediction": answer,
            "sample_time": end - start
        })

    # Save results
    results_dir = os.path.join(experiment_config.result_path, "icot_si")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.save_json(results, f"{results_dir}/inference_results.json")
    os.remove(f"{output_path}/adapter_config.json")
    os.remove(f"{output_path}/adapter_model.safetensors")
    return results