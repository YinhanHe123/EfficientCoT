import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.deepspeed_contemp_generator import PipelinedContemplationGenerator
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM
from utils.logging import Logger
import utils.utils as utils
import deepspeed

def compute_loss_ans(contemp_states, teacher_model, teacher_tokenizer, answer_loss_fn, combined_inputs, answer_inputs, exp_config, device, mode="train"):
    """
    Compute the answer loss using the contemplation states

    This function is largely the same as the original, but with better handling for device placement
    """
    # Determine whether to compute gradients based on mode
    context_manager = torch.enable_grad() if mode == "train" else torch.no_grad()

    with context_manager:
        # Get the total sequence length and limit contemp states to max_contemp_tokens
        contemp_len = min(contemp_states.size(1), exp_config.max_contemp_tokens)

        # Get the embeddings from the model's embedding layer (no gradients needed)
        with torch.no_grad():
            inputs_embeds = teacher_model.get_input_embeddings()(combined_inputs.input_ids)
            answer_embeds = teacher_model.get_input_embeddings()(answer_inputs.input_ids)

        # Keep contemp_states gradients in train mode
        contemp_states_to_use = contemp_states if mode == "train" else contemp_states.detach()

        # Create a new inputs_embeds by concatenating
        combined_embeds = torch.cat([
            inputs_embeds,
            contemp_states_to_use[:, -contemp_len:, :],
            answer_embeds
        ], dim=1)

        # Create proper attention mask and position ids
        attention_mask = torch.ones(
            (combined_inputs.input_ids.size(0), combined_embeds.shape[1]),
            dtype=torch.long,
            device=device
        )
        position_ids = torch.arange(
            combined_embeds.shape[1],
            dtype=torch.long,
            device=device
        ).unsqueeze(0).expand(combined_inputs.input_ids.size(0), -1)

        # Forward pass with combined embeddings - conditionally use no_grad
        if mode == "train":
            # This allows gradients to flow through the teacher model in train mode
            teacher_outputs = teacher_model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True
            )
        else:
            # In eval mode, use no_grad to save memory
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True
                )

        # Get logits from the teacher model
        logits = teacher_outputs.logits

        # Get answer labels (shifted by one token)
        answer_labels = answer_inputs.input_ids[:, 1:]  # Shifted by one token

        # Get the index to start predictions from (where the answer begins)
        # This is where the original input ends plus the contemplation tokens
        start_idx = combined_inputs.input_ids.size(1) + contemp_len - 1
        seq_length = answer_labels.size(1)

        # Get all relevant logits at once
        answer_logits = logits[:, start_idx:start_idx+seq_length, :]
        # Reshape for the loss function
        answer_logits_flat = answer_logits.reshape(-1, answer_logits.size(-1))
        answer_labels_flat = answer_labels.reshape(-1)

        # Calculate loss for all positions at once
        l_ans = answer_loss_fn(answer_logits_flat, answer_labels_flat)

    return l_ans


def train_contemplation_generator_with_deepspeed(
    model_config,
    exp_config,
    train_dataset,
    eval_dataset,
    local_rank,
    variation='vanilla',
    num_stages=2
):
    """
    Train a contemplation generator with DeepSpeed pipeline parallelism

    Args:
        model_config: Model configuration
        exp_config: Experiment configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        local_rank: Local rank for distributed training
        variation: Training variation ('vanilla', 'no_sentence_transformer', or 'no_l_reason')
        num_stages: Number of pipeline stages
    """
    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed()

    # Create the model
    model = PipelinedContemplationGenerator(
        model_config.student_model_name,
        model_config.teacher_model_name,
        model_config.teacher_hidden_dim,
        f"cuda:{local_rank}"
    )

    # Create pipelined model
    pipeline_model = model.create_pipeline(num_stages=num_stages)

    # Define DeepSpeed configuration
    ds_config = {
        "train_batch_size": exp_config.batch_size,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": exp_config.batch_size // 2,
        "steps_per_print": 10,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": exp_config.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": exp_config.weight_decay
            }
        },
        "fp16": {
            "enabled": False  # Set to True for mixed precision training
        },
        "zero_optimization": {
            "stage": 0  # Disable ZeRO when using Pipeline
        },
        "pipeline": {
            "stages": num_stages,
            "partition": "uniform",
            "seed_layers": True,
            "activation_checkpoint_interval": 0
        }
    }

    # Initialize DeepSpeed engine
    args = argparse.Namespace()
    args.local_rank = local_rank
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=[p for p in pipeline_model.parameters() if p.requires_grad],
        config=ds_config
    )

    # Initialize teacher model on the appropriate device
    if variation == 'vanilla' and model_engine.local_rank == 0:
        from models.deepspeed_sentence_transformer import PipelinedSentenceTransformer
        sentence_transformer = PipelinedSentenceTransformer.from_pretrained(
            f"{exp_config.model_save_path}/sentence_transformer"
        ).to(f"cuda:{model_engine.local_rank}")
        sentence_transformer.eval()

        # Make sure no gradients flow through the sentence transformer
        for param in sentence_transformer.parameters():
            param.requires_grad = False
    else:
        sentence_transformer = None

    # Load the teacher model on device 0 for computing loss
    if model_engine.local_rank == 0:
        teacher_model = LlamaForCausalLM.from_pretrained(model_config.teacher_model_name)
        teacher_model = teacher_model.to(f"cuda:{model_engine.local_rank}")
        teacher_model.eval()

        # Keep the teacher model in eval mode and don't compute gradients
        for param in teacher_model.parameters():
            param.requires_grad = False

        # Initialize teacher tokenizer for answer generation
        teacher_tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

        # Initialize answer loss function
        answer_loss_fn = nn.CrossEntropyLoss(ignore_index=teacher_tokenizer.pad_token_id)
    else:
        teacher_model = None
        teacher_tokenizer = None
        answer_loss_fn = None

    # Setup logger on rank 0
    if model_engine.global_rank == 0:
        logger = Logger(
            log_dir=exp_config.log_dir,
            experiment_name=f"ds_contemp_generator_{variation}"
        )
        logger.log_hyperparams({**exp_config.__dict__, **model_config.__dict__})

    # Create data loaders with proper distributed setup
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=model_engine.global_rank
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,  # Micro batch size
        sampler=train_sampler,
        num_workers=2
    )

    # Training loop
    best_val_loss = float('inf')
    best_state_dict = None

    for epoch in range(exp_config.num_epochs):
        # Set the epoch for the sampler
        train_sampler.set_epoch(epoch)

        # Training phase
        model_engine.train()
        total_loss = 0
        reason_loss = 0
        ans_loss = 0
        samples_processed = 0

        # Process data batches
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{exp_config.num_epochs} - Training")):
            # Only process on rank 0, then broadcast results
            if model_engine.local_rank == 0:
                # Process batch
                query = batch["query"]
                ground_truth_reasoning = batch["reasoning"]
                condensed_reasoning = batch["condensed_reasoning"]
                ground_truth_answer = batch["answer"]

                # Get teacher model's ground-truth reasoning hidden states
                with torch.no_grad():
                    # Tokenize ground truth reasoning
                    reason_inputs = teacher_tokenizer(
                        ground_truth_reasoning,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=exp_config.max_seq_length
                    ).to(f"cuda:{model_engine.local_rank}")

                    # Generate hidden states
                    tc_output = teacher_model(
                        reason_inputs.input_ids,
                        attention_mask=reason_inputs.attention_mask,
                        output_hidden_states=True
                    )

                    if variation == 'no_sentence_transformer':
                        gt_reason_hidden_states = tc_output.hidden_states[exp_config.start_layer_idx]
                    elif variation == 'no_l_reason':
                        pass
                    elif variation == 'vanilla':
                        # Get the required hidden states
                        gt_reason_hidden_states = tc_output.hidden_states[exp_config.start_layer_idx]

                        # Get reasoning embeddings from sentence transformer
                        gt_reason_embeddings = sentence_transformer(
                            gt_reason_hidden_states
                        )

                # Prepare input for the pipeline model
                query_condensed_reasoning = f"Question: {query[0]}\nAnswer: {condensed_reasoning[0]}"
                contemp_inputs = model.tokenizer(
                    query_condensed_reasoning,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=exp_config.max_seq_length
                ).to(f"cuda:{model_engine.local_rank}")

                # Broadcast input tensor
                torch.distributed.broadcast(contemp_inputs.input_ids, 0)
                if contemp_inputs.attention_mask is not None:
                    torch.distributed.broadcast(contemp_inputs.attention_mask, 0)
            else:
                # Placeholder tensors for other ranks
                contemp_inputs = None

            # Forward pass through the pipeline
            outputs = model_engine(contemp_inputs.input_ids if model_engine.local_rank == 0 else None)

            # Get the loss
            if model_engine.local_rank == 0 and outputs is not None:
                contemp_states = outputs

                # Compute reasoning loss if needed
                if variation == 'vanilla':
                    contemp_embeddings = sentence_transformer(contemp_states)

                    # Reasoning loss (1 - similarity)
                    similarity = sentence_transformer.compute_similarity(
                        gt_reason_embeddings, contemp_embeddings
                    )

                    l_reason = 1 - similarity.mean()
                elif variation == 'no_l_reason':
                    l_reason = torch.tensor(0.0, device=f"cuda:{model_engine.local_rank}")
                elif variation == 'no_sentence_transformer':
                    # cosine similarity of the hidden states
                    similarity = torch.nn.functional.cosine_similarity(
                        torch.mean(contemp_states, 1).squeeze(),
                        torch.mean(gt_reason_hidden_states, 1).squeeze(),
                        dim=-1
                    )
                    l_reason = 1 - similarity.mean()

                # Compute answer loss
                combined_input = f"Question: {query[0]}\nAnswer:"
                combined_inputs = teacher_tokenizer(
                    combined_input,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=exp_config.max_seq_length
                ).to(f"cuda:{model_engine.local_rank}")

                answer_inputs = teacher_tokenizer(
                    ground_truth_answer,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=exp_config.max_seq_length
                ).to(f"cuda:{model_engine.local_rank}")

                l_ans = compute_loss_ans(
                    contemp_states,
                    teacher_model,
                    teacher_tokenizer,
                    answer_loss_fn,
                    combined_inputs,
                    answer_inputs,
                    exp_config,
                    f"cuda:{model_engine.local_rank}",
                    mode="train"
                )

                # Combine losses
                if variation == 'no_l_reason':
                    loss = l_ans
                else:
                    loss = exp_config.alpha * l_reason + (1 - exp_config.alpha) * l_ans

                # Track loss statistics
                total_loss += loss.item() * contemp_inputs.input_ids.size(0)
                if variation != 'no_l_reason':
                    reason_loss += l_reason.item() * contemp_inputs.input_ids.size(0)
                ans_loss += l_ans.item() * contemp_inputs.input_ids.size(0)
                samples_processed += contemp_inputs.input_ids.size(0)

                # Broadcast loss to all ranks
                loss_tensor = torch.tensor([loss.item()], device=f"cuda:{model_engine.local_rank}")
            else:
                # Placeholder loss for other ranks
                loss_tensor = torch.tensor([0.0], device=f"cuda:{model_engine.local_rank}")

            # Broadcast loss
            torch.distributed.broadcast(loss_tensor, 0)

            # Backward and update
            model_engine.backward(loss_tensor.item())
            model_engine.step()

        # Calculate average losses
        if samples_processed > 0 and model_engine.global_rank == 0:
            avg_total_loss = total_loss / samples_processed
            avg_reason_loss = reason_loss / samples_processed if variation != 'no_l_reason' else 0
            avg_ans_loss = ans_loss / samples_processed

            # Log metrics
            logger.log_metrics({
                "total_loss": avg_total_loss,
                "reason_loss": avg_reason_loss,
                "ans_loss": avg_ans_loss
            }, epoch)

            print(f"Epoch {epoch+1} - Loss: {avg_total_loss:.4f} (Reason: {avg_reason_loss:.4f}, Ans: {avg_ans_loss:.4f})")

        # Synchronize all processes to ensure evaluation is done after training
        torch.distributed.barrier()

        # Evaluate the model periodically
        if (epoch + 1) % 5 == 0 and model_engine.global_rank == 0:
            eval_loss = evaluate_with_deepspeed(
                model,
                sentence_transformer if variation == 'vanilla' else None,
                teacher_model,
                teacher_tokenizer,
                eval_dataset,
                model_config,
                exp_config,
                variation
            )

            logger.log_metrics({"eval_loss": eval_loss}, epoch)
            print(f"Validation Loss: {eval_loss:.4f}")

            # Save best model
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                # Get a CPU copy of the model's state dict
                best_state_dict = {
                    k: v.cpu() for k, v in model.state_dict().items()
                }

        # Save checkpoint
        if (epoch + 1) % exp_config.save_interval == 0:
            checkpoint_path = os.path.join(exp_config.checkpoint_path, f"ds_contemp_generator_epoch{epoch+1}")
            client_state = {"checkpoint_step": epoch}
            model_engine.save_checkpoint(checkpoint_path, client_state=client_state)

    # Save the best model (only on rank 0)
    if model_engine.global_rank == 0 and best_state_dict is not None:
        # Create a non-pipelined version of the model for easier inference
        contemp_generator = PipelinedContemplationGenerator(
            model_config.student_model_name,
            model_config.teacher_model_name,
            model_config.teacher_hidden_dim,
            "cpu"  # Save initially to CPU
        )

        # Load the best state dict
        contemp_generator.load_state_dict(best_state_dict)

        # Save the model
        model_path = f"{exp_config.model_save_path}/contemp_generator"
        utils.create_directory(model_path)
        contemp_generator.save_pretrained(model_path)

        print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        logger.close()

    return None  # The model is saved to disk


def evaluate_with_deepspeed(model, sentence_transformer, teacher_model, teacher_tokenizer, eval_dataset, model_config, exp_config, variation='vanilla'):
    """
    Evaluate the model without using the pipeline (for simplicity)
    This function is designed to run on a single GPU (rank 0)
    """
    device = next(model.parameters()).device

    # Ensure the model is in evaluation mode
    model.eval()

    # Initialize answer loss function
    answer_loss_fn = nn.CrossEntropyLoss(ignore_index=teacher_tokenizer.pad_token_id)

    total_loss = 0
    reason_loss_sum = 0
    ans_loss_sum = 0
    samples_processed = 0

    # Create a simple data loader without distributed sampling
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=4,  # Larger batch size for evaluation
        shuffle=False
    )

    with torch.no_grad():  # No gradients for evaluation
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Process batch
            query = batch["query"]
            ground_truth_reasoning = batch["reasoning"]
            ground_truth_answer = batch["answer"]

            # Tokenize ground truth reasoning
            reason_inputs = teacher_tokenizer(
                ground_truth_reasoning,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=exp_config.max_seq_length
            ).to(device)

            # Get teacher model's hidden states
            tc_output = teacher_model(
                reason_inputs.input_ids,
                attention_mask=reason_inputs.attention_mask,
                output_hidden_states=True
            )

            # Get the hidden states from the specified layer
            gt_reason_hidden_states = tc_output.hidden_states[exp_config.start_layer_idx]

            # Generate contemplation tokens from student model
            query_inputs = model.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=exp_config.max_seq_length
            ).to(device)

            # Generate hidden states (use non-pipelined forward)
            contemp_states = model(
                query_inputs.input_ids,
                attention_mask=query_inputs.attention_mask
            )

            # Compute reasoning loss if needed
            if variation == 'vanilla':
                contemp_embeddings = sentence_transformer(contemp_states)
                gt_reason_embeddings = sentence_transformer(gt_reason_hidden_states)

                # Compute similarity for reasoning loss
                similarity = sentence_transformer.compute_similarity(
                    gt_reason_embeddings, contemp_embeddings
                )
                # Reasoning loss (1 - similarity)
                l_reason = 1 - similarity.mean()
            elif variation == 'no_sentence_transformer':
                similarity = torch.nn.functional.cosine_similarity(
                    torch.mean(contemp_states, 1).squeeze(),
                    torch.mean(gt_reason_hidden_states, 1).squeeze(),
                    dim=-1
                )
                l_reason = 1 - similarity.mean()
            elif variation == 'no_l_reason':
                l_reason = torch.tensor(0.0, device=device)

            # Compute answer loss
            combined_input = f"Question: {query[0]}\nAnswer:"
            combined_inputs = teacher_tokenizer(
                combined_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=exp_config.max_seq_length
            ).to(device)

            answer_inputs = teacher_tokenizer(
                ground_truth_answer,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=exp_config.max_seq_length
            ).to(device)

            # Compute answer loss with mode="eval"
            l_ans = compute_loss_ans(
                contemp_states,
                teacher_model,
                teacher_tokenizer,
                answer_loss_fn,
                combined_inputs,
                answer_inputs,
                exp_config,
                device,
                mode="eval"
            )

            # Combined loss
            if variation == 'no_l_reason':
                loss = l_ans
            else:
                loss = exp_config.alpha * l_reason + (1 - exp_config.alpha) * l_ans

            # Track losses
            batch_size = query_inputs.input_ids.size(0)
            total_loss += loss.item() * batch_size
            if variation != 'no_l_reason':
                reason_loss_sum += l_reason.item() * batch_size
            ans_loss_sum += l_ans.item() * batch_size
            samples_processed += batch_size

    # Calculate averages
    avg_total_loss = total_loss / samples_processed
    avg_reason_loss = reason_loss_sum / samples_processed if variation != 'no_l_reason' else 0
    avg_ans_loss = ans_loss_sum / samples_processed

    # Print detailed evaluation metrics
    print(f"Evaluation - Total Loss: {avg_total_loss:.4f}, Reason Loss: {avg_reason_loss:.4f}, Answer Loss: {avg_ans_loss:.4f}")

    return avg_total_loss