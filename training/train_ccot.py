import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import os
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset as HFDataset
import torch.nn.functional as F


def prepare_ccot_layer_dataset(queries, reasonings, tokenizer, device, layer_idx, compression_ratio=0.1, max_length=512):
    """
    Prepare a dataset for training a specific layer of the CCOT model using HuggingFace Dataset
    """
    # Precompute teacher hidden states
    teacher_model = AutoModelForCausalLM.from_pretrained(tokenizer.name_or_path)
    teacher_model.to(device)
    teacher_model.eval()
    
    # Prepare data dictionaries
    input_ids_list = []
    attention_mask_list = []
    target_states_list = []
    
    # Process each query-reasoning pair
    for query, reasoning in tqdm(zip(queries, reasonings), total=len(queries), desc="Preparing dataset"):
        # Tokenize the reasoning chain
        reasoning_inputs = tokenizer(reasoning, return_tensors="pt", padding="max_length", 
                                     truncation=True, max_length=max_length).to(device)
        
        # Get teacher hidden states
        with torch.no_grad():
            outputs = teacher_model(reasoning_inputs.input_ids, 
                                   attention_mask=reasoning_inputs.attention_mask,
                                   output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]
            
            # Select subset based on compression ratio
            target_states = select_subset_hidden_states(hidden_states, compression_ratio)
            
        # Tokenize the query
        query_inputs = tokenizer(query, return_tensors="pt", padding="max_length",
                                truncation=True, max_length=max_length).to(device)
        
        # Store the data
        input_ids_list.append(query_inputs.input_ids.squeeze().cpu().numpy())
        attention_mask_list.append(query_inputs.attention_mask.squeeze().cpu().numpy())
        target_states_list.append(target_states.squeeze().cpu().numpy())
    
    # Free up GPU memory
    del teacher_model
    torch.cuda.empty_cache()
    
    # Create HuggingFace Dataset
    dataset_dict = {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "target_states": target_states_list
    }
    
    return HFDataset.from_dict(dataset_dict)
# Utility functions
def hidden_state_loss(pred_states, target_states):
    """
    Compute a scaled mean squared error loss between hidden states
    
    Args:
        pred_states: Predicted hidden states
        target_states: Target hidden states
        
    Returns:
        Tensor: Loss value
    """
    # Compute the variance of target states for scaling
    target_variance = torch.var(target_states, dim=1, keepdim=True)
    
    # Compute MSE loss
    
    mse_loss = F.mse_loss(pred_states, target_states, reduction='none') / pred_states.shape[-1]
    
    # Scale by variance to normalize
    scaled_loss = mse_loss / (target_variance + 1e-6)
    
    return scaled_loss.mean()

def select_subset_hidden_states(hidden_states, ratio=0.1):
    """
    Select a subset of hidden states based on the compression ratio
    
    Args:
        hidden_states: Hidden states from the model
        ratio: Compression ratio to apply
        
    Returns:
        Tensor: Subset of hidden states
    """
    seq_len = hidden_states.shape[1]
    num_to_keep = max(1, int(seq_len * ratio))
    
    # Select indices at equal intervals
    indices = torch.linspace(0, seq_len-1, steps=num_to_keep, dtype=torch.long, device=hidden_states.device)
    
    # Extract the selected hidden states
    selected_states = hidden_states[:, indices, :]
    
    return selected_states

class CCOTLayerTrainer(Trainer):
    """Custom trainer for a specific layer of the CCOT model"""
    def __init__(self, model, args, train_dataset, eval_dataset=None, layer_idx=0, **kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs)
        self.layer_idx = layer_idx
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        target_states = inputs["target_states"]
        
        # print shapes for debugging
        
        # Get model outputs
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Extract the predicted hidden states
        if isinstance(outputs, tuple):
            # When the model returns a tuple (contemplation_states, hidden_states)
            predicted_states = outputs[0]
        else:
            # When the model directly returns contemplation_states
            predicted_states = outputs
            
        # Compute loss
        # print('predicted_states', predicted_states.shape)
        # print('target_states', target_states.shape)
        # temporary fix (implement the token selector later)
        target_states = target_states[:, :predicted_states.shape[1], :]
        loss = hidden_state_loss(predicted_states, target_states)
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override the evaluation loop to ensure the loss is properly added to metrics
        """
        # Call the parent evaluation loop
        output = super().evaluation_loop(
            dataloader, 
            description, 
            prediction_loss_only, 
            ignore_keys, 
            metric_key_prefix
        )
        
        # Make sure loss is in the metrics
        if "loss" in output.metrics:
            output.metrics[f"{metric_key_prefix}_loss"] = output.metrics.pop("loss")
            
        return output
    

def prepare_ccot_decode_dataset(queries, answers, ccot_model, tokenizer, device, max_length=512):
    """
    Prepare a dataset for training the decode model with HuggingFace Dataset
    """
    # Prepare data dictionaries
    input_ids_list = []
    attention_mask_list = []
    contemp_states_list = []
    labels_list = []
    
    # Process each query-answer pair
    for query, answer in tqdm(zip(queries, answers), total=len(queries), desc="Preparing decode dataset"):
        # Tokenize the query
        query_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding="max_length",
                                max_length=max_length).to(device)
        
        # Generate contemplation tokens
        with torch.no_grad():
            contemp_states = ccot_model(query_inputs.input_ids, 
                                      attention_mask=query_inputs.attention_mask)
        
        # Extract reasoning and final answer
        reasoning_parts = answer.split('####')
        reasoning = reasoning_parts[0].strip()

        # The final answer comes after ####
        final_answer = reasoning_parts[1].strip() if len(reasoning_parts) > 1 else ""

        
        # Tokenize the answer
        answer_inputs = tokenizer(final_answer, return_tensors="pt", truncation=True, padding="max_length", 
                                 max_length=5)
        
        # # Combine query and answer for label creation
        # combined = f"{query} {answer}"
        # combined_inputs = tokenizer(combined, return_tensors="pt", truncation=True, padding="max_length",
        #                            max_length=max_length).to(device)
        
        # # Create labels (shift by 1)
        # labels = combined_inputs.input_ids.clone()
        # labels[:, :query_inputs.input_ids.size(1)] = -100  # Mask out the query part
        
        # Store the data
        input_ids_list.append(query_inputs.input_ids.cpu().numpy())
        attention_mask_list.append(query_inputs.attention_mask.cpu().numpy())
        contemp_states_list.append(contemp_states.cpu().numpy())
        labels_list.append(answer_inputs.input_ids.squeeze().numpy())
    
    # Create HuggingFace Dataset
    dataset_dict = {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "contemp_states": contemp_states_list,
        "labels": labels_list
    }
    
    return HFDataset.from_dict(dataset_dict)

class CCOTDecodeTrainer(Trainer):
    """Custom trainer for the CCOT decode model"""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        contemp_states = inputs["contemp_states"]
        labels = inputs["labels"]
        
        print(f"input_ids shape: {input_ids.shape}")
        print(f"contemp_states shape: {contemp_states.shape}")
        print(f"labels shape: {labels.shape}")
        
        # Forward pass with contemplation states
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            contemp_states=contemp_states,
            labels=None
        )
        
        # Cross-entropy loss is computed inside the model
        # loss = outputs.loss
        # crop logits to be the same size as labels
        logits = outputs.logits[:, :labels.shape[1], :]  # crop logits to the same length as labels
        loss = F.cross_entropy(logits.transpose(1,2), labels, reduction='mean')
        return (loss, outputs) if return_outputs else loss

def train_ccot_model(
    base_model_name,
    train_dataset,
    eval_dataset,
    output_path,
    compression_ratio=0.1,
    autoregressive_layer=15,
    lora_rank=16,
    lora_alpha=32,
    learning_rate=1e-4,
    num_epochs_per_layer=2,
    batch_size=8,
    device="cuda",
    eval_steps=1
):
    """
    Train a CCoT model layer-by-layer using LoRA fine-tuning
    
    Args:
        base_model_name: Name of the base model
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        output_path: Path to save the model
        compression_ratio: Ratio of compressed tokens to full reasoning chain
        autoregressive_layer: Layer to use for autoregressive generation
        lora_rank: Rank for LoRA adaptation
        lora_alpha: Alpha parameter for LoRA
        learning_rate: Learning rate for optimization
        num_epochs_per_layer: Number of epochs for each layer training
        batch_size: Batch size for training
        device: Device to run the model on
        eval_steps: Number of steps between evaluations
        
    Returns:
        Trained CCoT model
    """
    from models.ccot_model import CCoTModel
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize the CCoT model
    ccot_model = CCoTModel(
        base_model_name,
        compression_ratio=compression_ratio,
        autoregressive_layer=autoregressive_layer,
        device=device
    )
    
    # Extract training data
    queries = [item["question"] for item in train_dataset.dataset]
    reasonings = [item["answer"] for item in train_dataset.dataset]
    # print(queries)
    # Extract evaluation data
    eval_queries = [item["question"] for item in eval_dataset.dataset]
    eval_reasonings = [item["answer"] for item in eval_dataset.dataset]
    
    # Determine the number of layers to train
    # We train from autoregressive_layer to the end
    num_layers = ccot_model.model.config.num_hidden_layers
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_path}/checkpoints",
        num_train_epochs=num_epochs_per_layer,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_dir=f"{output_path}/logs",
        logging_steps=10,
        label_names=['target_states'],
        load_best_model_at_end = True,
        save_strategy="steps",
        save_steps=eval_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        push_to_hub=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    # Train each layer starting from autoregressive_layer
    # for layer_idx in range(autoregressive_layer, num_layers):
    for layer_idx in range(autoregressive_layer, autoregressive_layer+1): # this line is for debugging
        print(f"===== Training layer {layer_idx} =====")
        
        # Apply LoRA to the current layer
        ccot_model.model = ccot_model.apply_lora_layer_by_layer(
            layer_idx=layer_idx,
            rank=lora_rank,
            alpha=lora_alpha
        )
        
        # Create datasets for this layer using the new function
        train_layer_dataset = prepare_ccot_layer_dataset(
            queries=queries,
            reasonings=reasonings,
            tokenizer=tokenizer,
            device=device,
            layer_idx=layer_idx,
            compression_ratio=compression_ratio
        )
        print('train_dataset_attributes', train_layer_dataset[0].keys())
        
        eval_layer_dataset = prepare_ccot_layer_dataset(
            queries=eval_queries,
            reasonings=eval_reasonings,
            tokenizer=tokenizer,
            device=device,
            layer_idx=layer_idx,
            compression_ratio=compression_ratio
        )
        print('eval_dataset_attributes', eval_layer_dataset[0].keys())
        
        # Create layer trainer
        trainer = CCOTLayerTrainer(
            model=ccot_model,
            args=training_args,
            train_dataset=train_layer_dataset,
            eval_dataset=eval_layer_dataset,
            layer_idx=layer_idx
        )
        
        # Train the layer
        trainer.train() # continue iterations will go from the parameters of the previously trained iteration
        # merge Lora parameters with original ccot to get new ccot mode parameters
        ccot_model.model = ccot_model.model.merge_and_unload(progressbar=True, safe_merge=True)
    # save the model 
    ccot_model.save_pretrained(output_path)
    return ccot_model

def train_ccot_decode_model(
    base_model_name,
    train_dataset,
    eval_dataset,
    ccot_model,
    output_path,
    lora_rank=16,
    lora_alpha=32,
    learning_rate=5e-5,
    num_epochs=5,
    batch_size=4,
    device="cuda",
    eval_steps=100
):
    """
    Train a decoder model to generate answers using contemplation tokens
    
    Args:
        base_model_name: Name of the base model
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        ccot_model: Trained CCoT model to generate contemplation tokens
        output_path: Path to save the model
        lora_rank: Rank for LoRA adaptation
        lora_alpha: Alpha parameter for LoRA
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to run the model on
        eval_steps: Number of steps between evaluations
        
    Returns:
        Trained decoder model
    """
    from models.ccot_model import CCOTDecodeModel
    
    # Create output directory
    
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize the decode model
    decode_model = CCOTDecodeModel(base_model_name, device=device)
    
    # Apply LoRA to the model
    decode_model = decode_model.apply_lora(rank=lora_rank, alpha=lora_alpha)
    
    # Extract training data
    queries = [item["question"] for item in train_dataset.dataset]
    answers = [item["answer"] for item in train_dataset.dataset]
    
    # Extract evaluation data
    eval_queries = [item["question"] for item in eval_dataset.dataset]
    eval_answers = [item["answer"] for item in eval_dataset.dataset]
    
    # Create datasets
    # Create datasets using the new function
    train_decode_dataset = prepare_ccot_decode_dataset(
        queries=queries,
        answers=answers,
        ccot_model=ccot_model,
        tokenizer=tokenizer,
        device=device
    )

    eval_decode_dataset = prepare_ccot_decode_dataset(
        queries=eval_queries,
        answers=eval_answers,
        ccot_model=ccot_model,
        tokenizer=tokenizer,
        device=device
    )
    # delete ccot model and clean gpu
    del ccot_model
    torch.cuda.empty_cache()
    
    # Training arguments
    os.makedirs(output_path, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=f"{output_path}/checkpoints",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_dir=f"{output_path}/logs",
        logging_steps=10,
        label_names=['target_states'],
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=eval_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        push_to_hub=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    # Create trainer
    trainer = CCOTDecodeTrainer(
        model=decode_model,
        args=training_args,
        train_dataset=train_decode_dataset,
        eval_dataset=eval_decode_dataset
    )
    # print('\n\n\n-------------------------label_names---------------------\n\n\n',trainer.args.label_names)
    # Train the model
    trainer.train()
    decode_model = decode_model.merge_and_unload(progressbar=True, safe_merge=True)
    # Save the model
    decode_model.save_pretrained(output_path)
    return decode_model