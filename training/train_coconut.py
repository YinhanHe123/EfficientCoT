import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
from models.coconut_model import CoconutModel
from peft import get_peft_model, LoraConfig, TaskType

def train_coconut_model(
    base_model_name,
    train_dataset,
    eval_dataset,
    output_path,
    learning_rate=1e-5,
    num_epochs=5,
    batch_size=1,
    max_continuous_tokens=5,
    device="cuda"
):
    """
    Train a Coconut model with multi-stage curriculum learning as described in the paper.

    Args:
        base_model_name: Name of the base LLM model
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        output_path: Path to save the model
        learning_rate: Learning rate for optimization
        num_epochs: Number of epochs for each stage
        batch_size: Batch size for training
        max_continuous_tokens: Maximum number of continuous thought tokens
        device: Device to run the model on

    Returns:
        Trained Coconut model
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Initialize the Coconut model
    coconut_model = CoconutModel(base_model_name, device=device)

    # Apply LoRA to make training more efficient
    target_modules = []
    # Add key layers for LoRA adaptation
    for i in range(len(coconut_model.model.model.layers)):
        target_modules.extend([
            f"model.layers.{i}.self_attn.q_proj",
            f"model.layers.{i}.self_attn.k_proj",
            f"model.layers.{i}.self_attn.v_proj",
            f"model.layers.{i}.self_attn.o_proj"
        ])

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,  # LoRA rank
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=target_modules,
    )

    # Enable input requires grad
    if hasattr(coconut_model.model, 'enable_input_require_grads'):
        coconut_model.model.enable_input_require_grads()

    coconut_model.model = get_peft_model(coconut_model.model, peft_config)
    coconut_model.model.model.lm_head.weight.requires_grad = True
    coconut_model.model.model.model.embed_tokens.weight.requires_grad=True

    def freeze_old_weights_hook(grad):
        return torch.nan_to_num(grad, nan=0, posinf=0, neginf=0) * torch.concat([torch.zeros_like(grad[:-1]), torch.ones_like(grad[-1:])], dim=0).to(grad.device)

    lm_head_hook = coconut_model.model.model.lm_head.weight.register_hook(freeze_old_weights_hook)
    embed_tokens_hook = coconut_model.model.model.model.embed_tokens.weight.register_hook(freeze_old_weights_hook)

    # Extract training data
    queries = [item["query"] for item in train_dataset]
    reasonings = [item["full_answer"] for item in train_dataset]

    # Extract evaluation data
    eval_queries = [item["query"] for item in eval_dataset]
    eval_reasonings = [item["full_answer"] for item in eval_dataset]

    # Multi-stage curriculum learning
    num_stages = 4  # Initial stage + 3 stages with continuous thoughts
    best_loss = float('inf')
    best_model_state = None

    for stage in range(num_stages):
        print(f"=== Stage {stage+1}/{num_stages} ===")

        # Prepare data for this stage
        train_data = prepare_stage_data(
            coconut_model.tokenizer,
            queries,
            reasonings,
            stage,
            max_continuous_tokens
        )

        eval_data = prepare_stage_data(
            coconut_model.tokenizer,
            eval_queries,
            eval_reasonings,
            stage,
            max_continuous_tokens
        )

        # Create data loaders
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )

        eval_loader = DataLoader(
            eval_data,
            batch_size=batch_size,
            shuffle=False
        )

        # Reset optimizer for each stage
        optimizer = torch.optim.AdamW(
            coconut_model.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Training loop for this stage
        for epoch in range(num_epochs):
            # Training
            coconut_model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Stage {stage+1}, Epoch {epoch+1}/{num_epochs} - Training"):
                # Unpack batch
                input_ids, attention_mask, labels = [b.to(device) for b in batch]

                # Reset gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = coconut_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # Compute loss
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Stage {stage+1}, Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}")

            # Evaluation
            coconut_model.eval()
            eval_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"Stage {stage+1}, Epoch {epoch+1}/{num_epochs} - Evaluation"):
                    # Unpack batch
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]

                    # Forward pass
                    outputs = coconut_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    # Compute loss
                    loss = outputs.loss
                    eval_loss += loss.item()

            avg_eval_loss = eval_loss / len(eval_loader)
            print(f"Stage {stage+1}, Epoch {epoch+1}/{num_epochs} - Eval Loss: {avg_eval_loss:.6f}")

            # Save best model
            if avg_eval_loss < best_loss:
                best_loss = avg_eval_loss

                # Save best model state
                if hasattr(coconut_model.model, 'merge_and_unload'):
                    # For LoRA models, merge and save
                    best_model = coconut_model.model.merge_and_unload()
                    coconut_model.model = best_model

                # Save checkpoint
                checkpoint_path = os.path.join(output_path, f"stage_{stage+1}_best.pt")
                torch.save(coconut_model.state_dict(), checkpoint_path)
                print(f"Saved best model with eval loss: {best_loss:.6f}")

    # Save the final model
    if hasattr(coconut_model.model, 'merge_and_unload'):
        # For LoRA models, merge weights
        coconut_model.model = coconut_model.model.merge_and_unload()

    coconut_model.save_pretrained(output_path)
    print(f"Coconut model saved to {output_path}")
    lm_head_hook.remove()
    embed_tokens_hook.remove()
    return coconut_model

def prepare_stage_data(tokenizer, queries, reasonings, stage, max_continuous_tokens):
    """
    Prepare data for a specific training stage in the curriculum.

    Args:
        tokenizer: Tokenizer for the model
        queries: List of queries
        reasonings: List of reasoning chains
        stage: Current curriculum stage (0 = initial, 1+ = later stages)
        max_continuous_tokens: Maximum number of continuous thought tokens

    Returns:
        TensorDataset for this stage
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    bot_token = "<bot>"
    eot_token = "<eot>"

    for query, reasoning in zip(queries, reasonings):
        # Parse the reasoning chain
        if "####" in reasoning:
            reasoning_parts = reasoning.split("####")
            reasoning_chain = reasoning_parts[0].strip()
            answer = reasoning_parts[1].strip() if len(reasoning_parts) > 1 else ""
        else:
            reasoning_chain = reasoning
            answer = ""

        # Split reasoning into steps
        reasoning_steps = [step.strip() for step in reasoning_chain.split('\n') if step.strip()]

        if stage == 0:
            # Initial stage: standard CoT
            input_text = f"{query} {bot_token} {eot_token} {' '.join(reasoning_steps)} {answer}"
        else:
            # Later stages: replace first 'stage' steps with continuous thoughts
            steps_to_remove = min(stage, len(reasoning_steps))
            remaining_steps = reasoning_steps[steps_to_remove:]

            # Add continuous thoughts
            continuous_thoughts = f"{bot_token} {' '.join(['<thought>'] * (steps_to_remove * max_continuous_tokens))} {eot_token}"

            # Combine for input
            input_text = f"{query} {continuous_thoughts} {' '.join(remaining_steps)} {answer}"

        # Tokenize
        encodings = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()

        # Create labels (shift right, mask out query and continuous thoughts)
        labels = input_ids.clone()

        # Mask out the query part before <bot>
        bot_pos = (input_ids == tokenizer.convert_tokens_to_ids(bot_token)).nonzero(as_tuple=True)[0]
        if len(bot_pos) > 0:
            labels[:bot_pos[0]] = -100

        # Mask out continuous thoughts between <bot> and <eot>
        eot_pos = (input_ids == tokenizer.convert_tokens_to_ids(eot_token)).nonzero(as_tuple=True)[0]
        if len(bot_pos) > 0 and len(eot_pos) > 0:
            labels[bot_pos[0]:eot_pos[0]+1] = -100

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    # Create tensor dataset
    return TensorDataset(
        torch.stack(input_ids_list),
        torch.stack(attention_mask_list),
        torch.stack(labels_list)
    )

def run_inference_with_coconut(coconut_model, dataset, config):
    """
    Run inference with the trained Coconut model

    Args:
        coconut_model: Trained Coconut model
        dataset: Dataset for evaluation
        config: Configuration parameters

    Returns:
        List of results
    """
    device = config.device
    coconut_model = coconut_model.to(device)
    coconut_model.eval()

    results = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Running inference with Coconut"):
            query = sample["query"]

            # Generate answer with continuous thoughts
            answer = coconut_model.generate_with_continuous_thoughts(
                query,
                max_continuous_tokens=config.eval_max_contemp_tokens,
                max_new_tokens=100
            )

            results.append({
                "query": query,
                "ground_truth": sample.get("answer", ""),
                "prediction": answer
            })

    # Save results
    result_dir = f"{config.result_path}/coconut"
    os.makedirs(result_dir, exist_ok=True)

    import json
    with open(f"{result_dir}/inference_results.json", "w") as f:
        json.dump({"results": results}, f, indent=2)

    return results