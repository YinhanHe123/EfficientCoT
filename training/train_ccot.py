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
from datasets import load_from_disk
import torch.nn.functional as F
import openai
import gc


def prepare_ccot_layer_dataset(queries, reasonings, tokenizer, device, layer_idx, compression_ratio=0.1, max_length=120):
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
    #

    # Process each query-reasoning pair
    for query, reasoning in tqdm(zip(queries, reasonings), total=len(queries), desc="Preparing dataset"):
        # Tokenize the reasoning chain
        # seelct important tokens as target target tokens in reasoning.
        num_of_compressed_tokens = int(max_length * compression_ratio)
        openai.api_key = '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' # Replace with your OpenAI API key
        selected_indices = []
        repeat = 0
        while len(selected_indices) < num_of_compressed_tokens:
            if repeat > 3:
                print('openai api error! generated indices multiple times but still not enough indices, use even-spaced selections')
                selected_indices = np.arange(0, len(reasoning.split()), step=max(1, len(reasoning.split()) // num_of_compressed_tokens))
                break
            selected_indices = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user",
                "content": f"You should select {num_of_compressed_tokens} important words from the following text. I need you to only output the words' indices increasingly, DO NOT OUTPUT ANYTHING ELSE. For example, for text 'Where is my pencil?', the important words are 'Where' and 'pencil' at position 0 and 4, so the output is '0, 4'. Now, do it for the following text: "+reasoning}
            ]
            ).choices[0].message.content
            selected_indices = torch.tensor(list(map(int, selected_indices.split(','))))
            repeat += 1
        selected_indices = selected_indices[:num_of_compressed_tokens]
        # print('selected_indices:', selected_indices)

        # Tokenize the reasoning
        reasoning_inputs = tokenizer(reasoning, return_tensors="pt", padding="max_length",
                                     truncation=True, max_length=max_length).to(device)

        # Get teacher hidden states
        with torch.no_grad():
            outputs = teacher_model(reasoning_inputs.input_ids,
                                   attention_mask=reasoning_inputs.attention_mask,
                                   output_hidden_states=True)
            hidden_states = outputs.hidden_states[15]

            # Select subset based on compression ratio
            target_states = hidden_states[:, selected_indices, :]

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
    as described in the CCoT paper (Section 4.1).

    LOSSφ(zl_i, ẑl_i) = (1/k) * Σ_(i=1 to k) [1/σ²(zl_i) * MSE(zl_i, ẑl_i)]

    Args:
        pred_states: Predicted hidden states [batch_size, num_tokens, hidden_dim]
        target_states: Target hidden states [batch_size, num_tokens, hidden_dim]

    Returns:
        Tensor: Loss value
    """
    # Number of tokens (k in the paper's notation)
    k = pred_states.shape[1]

    # Initialize total loss
    total_loss = 0.0

    # Process each token position independently as per the paper formula
    for i in range(k):
        # Get the hidden states for the current token position
        z_i = target_states[:, i, :]  # [batch_size, hidden_dim]
        z_hat_i = pred_states[:, i, :]  # [batch_size, hidden_dim]

        # Compute variance of target states for each sample in the batch
        # This computes σ²(zl_i) along the hidden dimension
        variance = torch.var(z_i, dim=1, keepdim=True)  # [batch_size, 1]

        # Compute Mean Squared Error
        mse = torch.mean((z_i - z_hat_i) ** 2, dim=1, keepdim=True)  # [batch_size, 1]

        # Scale MSE by variance
        scaled_mse = mse / (variance + 1e-6)  # Add epsilon for numerical stability

        # Sum the scaled MSE for this token position
        total_loss += scaled_mse.mean()

    # Average over all token positions
    return total_loss / k


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


def prepare_ccot_decode_dataset(queries, answers, ccot_model, tokenizer, device, max_length=120):
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
    lora_rank=128,
    lora_alpha=256,
    learning_rate=1e-4,
    num_epochs_per_layer=10,
    batch_size=8,
    device="cuda",
    eval_steps=5
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
        evaluation_strategy="epoch",
        eval_steps=eval_steps,
        logging_dir=f"{output_path}/logs",
        logging_steps=10,
        label_names=['target_states'],
        load_best_model_at_end = True,
        save_strategy="best",
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        push_to_hub=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    # Train each layer starting from autoregressive_layer
    for layer_idx in range(0, num_layers): # train every three layers
    # for layer_idx in range(autoregressive_layer, autoregressive_layer+2): # this line is for debugging
        print(f"===== Training layer {layer_idx} =====")

        # Apply LoRA to the current layer
        ccot_model.model = ccot_model.apply_lora_layer_by_layer(
            layer_idx=layer_idx,
            rank=lora_rank,
            alpha=lora_alpha
        )
        # if train_layer_dataset is there, load, if not, prepare
        if os.path.exists(f"{output_path}/layer_train_dataset.json"):
            train_layer_dataset = load_from_disk(f"{output_path}/layer_train_dataset.json")
        else:
            # Create datasets for this layer using the new function
            train_layer_dataset = prepare_ccot_layer_dataset(
                queries=queries,
                reasonings=reasonings,
                tokenizer=tokenizer,
                device=device,
                layer_idx=layer_idx,
                compression_ratio=compression_ratio
            )
            train_layer_dataset.save_to_disk(f"{output_path}/layer_train_dataset.json")
        print('train_dataset_attributes', train_layer_dataset[0].keys())
        if os.path.exists(f"{output_path}/layer_eval_dataset.json"):
            eval_layer_dataset = load_from_disk(f"{output_path}/layer_eval_dataset.json")
        else:
            eval_layer_dataset = prepare_ccot_layer_dataset(
                queries=eval_queries,
                reasonings=eval_reasonings,
                tokenizer=tokenizer,
                device=device,
                layer_idx=layer_idx,
                compression_ratio=compression_ratio
            )
            eval_layer_dataset.save_to_disk(f"{output_path}/layer_eval_dataset.json")
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
        os.system(f"rm -r {output_path}/checkpoints")

        gc.collect()
        # Free up GPU memory
        torch.cuda.empty_cache()

    # Save the model after layer training
    temp_model_save_path = f"{output_path}/model_before_end_predictor"
    os.makedirs(temp_model_save_path, exist_ok=True)
    ccot_model.save_pretrained(temp_model_save_path)
    # delete the saved checkpoints


    # Train the END_psi module as a separate step
    print("===== Training END_psi module =====")
    ccot_model = train_end_predictor(
        ccot_model=ccot_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_path=output_path,
        learning_rate=learning_rate / 10,  # Use lower learning rate for end predictor
        num_epochs=3,
        batch_size=batch_size,
        device=device
    )

    # Save the final model
    ccot_model.save_pretrained(output_path)
    return ccot_model

def train_ccot_decode_model(
    base_model_name,
    train_decode_dataset,
    eval_decode_dataset,
    output_path,
    lora_rank=64,
    lora_alpha=128,
    learning_rate=5e-5,
    num_epochs=15,
    batch_size=4,
    device="cuda",
    eval_steps=25
):
    """
    Train a decoder model to generate answers using contemplation tokens

    Args:
        base_model_name: Name of the base model
        train_decode_dataset: Dataset for training
        eval_decode_dataset: Dataset for evaluation
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
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Initialize the decode model
    decode_model = CCOTDecodeModel(base_model_name, device=device)

    # Apply LoRA to the model
    decode_model = decode_model.apply_lora(rank=lora_rank, alpha=lora_alpha)

    # Training arguments
    os.makedirs(output_path, exist_ok=True)
    print(output_path)
    os.makedirs(f"{output_path}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_path}/logs", exist_ok=True)
    training_args = TrainingArguments(
        output_dir=f"{output_path}/checkpoints",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        eval_steps=eval_steps,
        logging_dir=f"{output_path}/logs",
        logging_steps=10,
        label_names=['labels'],
        load_best_model_at_end=True,
        save_strategy="best",
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
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
    decode_model.model = decode_model.model.merge_and_unload(progressbar=True, safe_merge=True)
    # Save the model
    decode_model.save_pretrained(output_path)
    os.system(f"rm -r {output_path}/checkpoints")

    return decode_model


def train_end_predictor(
    ccot_model,
    train_dataset,
    eval_dataset,
    output_path,
    learning_rate=1e-5,
    num_epochs=5,
    batch_size=8,
    device="cuda"
):
    """
    Train the END_psi module, which is a binary classifier that predicts
    when to stop generating contemplation tokens.

    As clarified by the author: "The END_psi module is a simple binary classifier
    on the last layer hidden state of the generated responses. This is trained
    separately from DECODE (crucially when the DECODE_psi weights are frozen)."

    Args:
        ccot_model: Trained CCOT model
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        output_path: Path to save the model
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to run the model on

    Returns:
        Trained CCOT model with END_psi module updated
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    print("=== Training END_psi Module ===")

    # Freeze all parameters except the end predictor
    for param in ccot_model.parameters():
        param.requires_grad = False

    # Only enable gradients for the end_predictor
    for param in ccot_model.end_predictor.parameters():
        param.requires_grad = True

    # Extract queries for generating contemplation tokens
    queries = [item["question"] for item in train_dataset.dataset]
    reasonings = [item["answer"] for item in train_dataset.dataset]

    # Tokenizer for processing inputs
    tokenizer = ccot_model.tokenizer

    # Prepare training data for the end predictor
    end_pred_inputs = []
    end_pred_labels = []

    for query, reasoning in tqdm(zip(queries, reasonings), total=len(queries), desc="Preparing END_psi training data"):
        # Tokenize query
        query_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding="max_length",
                                max_length=120).to(device)

        # Tokenize reasoning to determine target sequence length
        reasoning_inputs = tokenizer(reasoning, return_tensors="pt").to(device)
        target_length = min(int(reasoning_inputs.input_ids.size(1) * ccot_model.compression_ratio), 50)

        # Generate contemplation tokens with gradient tracking for the end predictor
        ccot_model.eval()  # Set model to eval mode but with end predictor gradients enabled

        # Forward pass to generate complete sequence of contemplation tokens
        with torch.no_grad():  # No need for gradients here, just generating training data
            contemplation_states, _ = ccot_model(
                query_inputs.input_ids,
                attention_mask=query_inputs.attention_mask,
                output_hidden_states=True
            )

        # For each position, create training samples for the end predictor
        max_pos = min(contemplation_states.size(1), target_length + 5)  # Add a few extra positions

        for pos in range(1, max_pos):
            # Use the hidden state at this position
            hidden_state = contemplation_states[0, pos-1]  # Previous token's hidden state

            # Label: 1 if we should stop (position >= target_length), 0 otherwise
            label = 1.0 if pos >= target_length else 0.0

            end_pred_inputs.append(hidden_state.cpu().numpy())
            end_pred_labels.append(label)

    # Convert to PyTorch tensors
    end_pred_inputs = torch.tensor(end_pred_inputs, dtype=torch.float32)
    end_pred_labels = torch.tensor(end_pred_labels, dtype=torch.float32).unsqueeze(1)

    # Create dataset and dataloader
    end_pred_dataset = TensorDataset(end_pred_inputs, end_pred_labels)
    end_pred_loader = DataLoader(end_pred_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer for end predictor only
    optimizer = optim.AdamW(ccot_model.end_predictor.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    ccot_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in tqdm(end_pred_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            logits = ccot_model.end_predictor(inputs)

            # Compute loss
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct_preds += (predictions == labels).sum().item()
            total_preds += labels.size(0)

        # Epoch summary
        epoch_loss = epoch_loss / len(end_pred_dataset)
        accuracy = correct_preds / total_preds
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the model with updated end predictor
    ccot_model.save_pretrained(output_path)

    print("END_psi module training completed!")
    return ccot_model