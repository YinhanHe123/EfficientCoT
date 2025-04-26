import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import time
import math
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteriaList, LogitsProcessorList
from peft import get_peft_model, LoraConfig, TaskType

# Add necessary imports from the existing project
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

def extract_answer(text):
    """Extract the final answer from text with '####' delimiter"""
    split_pattern = '####'
    if split_pattern not in text:
        return text.strip()
    else:
        _, ans = text.strip().split('####', 1)
        ans = ans.strip()
        return ans

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

    def prepare_batch(self, item, scheduled_to_remove=0):
        """Prepare a batch for training with CoT removal strategy"""
        query = item["query"][0]
        full_answer = item["full_answer"][0] if "full_answer" in item else f"{item['reasoning'][0]} #### {item['answer'][0]}"

        # Format input with query and full answer (reasoning + answer)
        input_text = f"Question: {query}\n{full_answer}"

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.device)

        input_ids = inputs.input_ids
        labels = input_ids.clone()

        # Find separator positions (question|reasoning|answer)
        question_end = get_sep_position(input_ids, self.tokenizer.encode("\n", add_special_tokens=False)[-1])
        reasoning_delimiter = get_sep_position(input_ids, self.tokenizer.encode("####", add_special_tokens=False)[-1])

        # Mask out question tokens in labels (we don't want to predict the question)
        for i in range(input_ids.shape[0]):
            labels[i, :question_end[i]+1] = self.tokenizer.pad_token_id

        # Apply CoT removal if scheduled
        if scheduled_to_remove > 0:
            batch_size = input_ids.shape[0]
            position_ids = None

            if self.keep_position:
                position_ids = torch.arange(0, input_ids.shape[1], device=self.device).unsqueeze(0).repeat(batch_size, 1)

            input_ids_tmp = []
            labels_tmp = []

            # Sample random removal offsets
            random_removal_offset = torch.multinomial(
                self.lambda_distribution,
                batch_size,
                replacement=True
            ).to(self.device)

            to_remove = scheduled_to_remove + random_removal_offset

            if self.removal_side == 'left':
                # Remove from start of reasoning
                removal_from = question_end + 1
                removal_to = question_end + 1 + to_remove
            else:
                # Remove from end of reasoning
                removal_to = reasoning_delimiter
                removal_from = reasoning_delimiter - to_remove

            for batch_id in range(batch_size):
                qe = question_end[batch_id]
                rd = reasoning_delimiter[batch_id] if reasoning_delimiter[batch_id] > 0 else input_ids.shape[1]-1

                # Calculate actual removal positions
                r_from = max(removal_from[batch_id], qe+1)
                r_to = min(removal_to[batch_id], rd)

                if self.keep_position and position_ids is not None:
                    position_ids[batch_id, r_from:] += r_to - r_from

                # Create new sequences with removed tokens
                input_ids_tmp.append(torch.cat([
                    input_ids[batch_id, :r_from],
                    input_ids[batch_id, r_to:]
                ]))

                labels_tmp.append(torch.cat([
                    labels[batch_id, :r_from],
                    labels[batch_id, r_to:]
                ]))

            # Batch padded sequences
            input_ids = batch_ids(input_ids_tmp, self.tokenizer.pad_token_id, self.device, input_ids.dtype)
            labels = batch_ids(labels_tmp, self.tokenizer.pad_token_id, self.device, labels.dtype)

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
        total_loss = 0
        steps = 0

        # Create dataloader
        dataloader = DataLoader(
            [{k: v for k,v  in d.items() if k != "condensed_reasoning"} for d in self.train_dataset],
            batch_size=self.batch_size,
            shuffle=True
        )

        for item in tqdm(dataloader, desc=f"Training (remove={scheduled_to_remove})"):
            batch = self.prepare_batch(item, scheduled_to_remove)
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                labels=batch['labels'],
                position_ids=batch['position_ids']
            )

            loss = outputs.loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

            if steps % 100 == 0:
                print(f"Step {steps} - Loss: {loss.item():.4f}")

        return total_loss / steps

    def evaluate(self, scheduled_to_remove=0):
        """Evaluate the model on the evaluation dataset"""
        self.model.eval()
        total_correct = 0
        total_samples = 0

        dataloader = DataLoader(
            [{k: v for k,v  in d.items() if k != "condensed_reasoning"} for d in self.eval_dataset],
            batch_size=self.batch_size,
            shuffle=False
        )

        with torch.no_grad():
            for item in tqdm(dataloader, desc=f"Evaluating (remove={scheduled_to_remove})"):
                batch = self.prepare_batch(item, scheduled_to_remove)

                # Get queries for generation
                queries = [f"Question: {q}\n" for q in item["query"]]
                query_inputs = self.tokenizer(
                    queries,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length
                ).to(self.device)

                # Generate answers
                stopping_criteria = DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)
                logits_processor = DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)

                generation_config = GenerationConfig.from_model_config(self.model.config)
                # generation_config.eos_token_id = -1  # Disable default EOS stopping

                outputs = self.model.generate(
                    input_ids=query_inputs.input_ids,
                    attention_mask=query_inputs.attention_mask,
                    position_ids=batch.get('position_ids', None),
                    max_new_tokens=150,
                    stopping_criteria=[stopping_criteria],
                    logits_processor=[logits_processor],
                    generation_config=generation_config,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

                # Compare generated answers with ground truth
                for i, output in enumerate(outputs):
                    generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    ground_truth = item["answer"][i]

                    generated_answer = extract_answer(generated_text)

                    if generated_answer.strip() == ground_truth.strip():
                        total_correct += 1

                    total_samples += 1

        accuracy = total_correct / total_samples
        return accuracy

    def train(self):
        """Train the model with progressive CoT removal"""
        best_accuracy = -1
        best_checkpoint = None
        scheduled_to_remove = 0

        steps_per_epoch = len(self.train_dataset) // self.batch_size
        steps_per_removed_token = max(1, int(steps_per_epoch / self.remove_per_epoch))

        for epoch in range(self.epochs):
            # Update scheduled_to_remove
            if epoch > 0:
                scheduled_to_remove += self.remove_per_epoch

            print(f"Epoch {epoch+1}/{self.epochs} - Removing {scheduled_to_remove} tokens")

            # Train for one epoch
            avg_loss = self.train_epoch(scheduled_to_remove)
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
        return best_checkpoint

def run_icot_si_baseline(train_dataset, eval_dataset, model_config, experiment_config):
    """
    Implicit Chain of Thought baseline
    Based on: "Implicit Chain of Thought Reasoning via Knowledge Distillation"

    Args:
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        model_config: Model configuration
        experiment_config: Experiment configuration

    Returns:
        List of prediction results
    """
    device = experiment_config.device

    # Set up output paths
    output_path = experiment_config.model_save_path

    # Check if the model has already been trained
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print("Training Implicit CoT model...")

        # Initialize trainer
        trainer = ImplicitCoTTrainer(
            model_name=model_config.teacher_model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_path=output_path,
            device=device,
            epochs=experiment_config.contemp_gen_epochs,
            lr=experiment_config.contemp_gen_lr,
            batch_size=1,
            max_seq_length=experiment_config.max_seq_length,
            remove_per_epoch=8,  # Gradually increase removal tokens
            removal_smoothing_lambda=0.5,  # Allow some variance in removal
            removal_side='left',  # Remove from start of reasoning
            keep_position=True,  # Maintain position embeddings
        )

        # Train the model
        trainer.train()

    # Load the trained model for inference
    tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    model = AutoModelForCausalLM.from_pretrained(output_path)
    model = model.to(device)
    model.eval()

    # Prepare inference results
    results = []

    # Run inference on evaluation dataset
    for item in tqdm(eval_dataset, desc="Running Implicit CoT inference"):
        query = item["query"]

        # Format input
        input_text = f"Question: {query}\n"

        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=experiment_config.max_seq_length
        ).to(device)

        # Set up generation parameters
        stopping_criteria = DoubleEOSStoppingCriteria(tokenizer.eos_token_id)
        logits_processor = DoubleEOSLogitsProcessor(tokenizer.eos_token_id)

        generation_config = GenerationConfig.from_model_config(model.config)
        # generation_config.eos_token_id = -1  # Disable default EOS stopping

        # Generate answer
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                stopping_criteria=[stopping_criteria],
                logits_processor=[logits_processor],
                generation_config=generation_config,
                do_sample=True,
                temperature=experiment_config.eval_temp,
                top_p=0.9
            )

        # Decode and extract answer
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = extract_answer(generated_text)

        # Store result
        results.append({
            "query": query,
            "ground_truth": item.get("answer", ""),
            "prediction": answer
        })

    # Save results
    results_dir = os.path.join(experiment_config.result_path, "icot_si")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    utils.save_json(results, f"{results_dir}/inference_results.json")

    return results