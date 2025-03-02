import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM
import utils.utils as utils
from utils.logging import Logger
from utils.distributed import convert_model_to_ddp, is_main_process, reduce_loss
import torch.distributed as dist
from training.train_contemp_gen import compute_loss_ans

def train_contemplation_generator_distributed(
    contemp_generator,
    sentence_transformer,
    train_dataset,
    eval_dataset,
    model_config,
    exp_config,
    variation,
    rank,
    world_size
):
    device = rank  # Use the current GPU rank as the device

    # Convert models to DDP format for distributed training
    contemp_generator = convert_model_to_ddp(contemp_generator, device)

    # Ensure sentence transformer is in evaluation mode and on the right device
    if variation == 'vanilla':
        if sentence_transformer is not None:
            sentence_transformer = sentence_transformer.to(device)
            sentence_transformer.eval()
            for param in sentence_transformer.parameters():
                param.requires_grad = False

    # Initialize teacher model
    teacher_model = LlamaForCausalLM.from_pretrained(model_config.teacher_model_name)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Set to evaluation mode
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Initialize teacher tokenizer for answer generation
    teacher_tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Initialize answer loss function
    answer_loss_fn = nn.CrossEntropyLoss(ignore_index=teacher_tokenizer.pad_token_id)

    # Define optimizers
    optimizer = optim.AdamW(
        contemp_generator.parameters(),
        lr=exp_config.learning_rate,
        weight_decay=exp_config.weight_decay
    )

    # Setup logger (only on main process)
    if is_main_process(rank):
        logger = Logger(
            log_dir=exp_config.log_dir,
            experiment_name=f"contemp_generator"
        )
        logger.log_hyperparams(exp_config.__dict__ | model_config.__dict__)

    # Create DistributedSampler for the datasets
    from torch.utils.data import DistributedSampler, DataLoader

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=exp_config.seed
    )

    # Create DataLoader with the samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=exp_config.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    if eval_dataset:
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=exp_config.batch_size,
            sampler=eval_sampler,
            num_workers=4,
            pin_memory=True
        )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(exp_config.num_epochs):
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)

        # Set contemp_generator to train mode only during training
        contemp_generator.train()

        total_loss = 0
        reason_loss = 0
        ans_loss = 0

        # Create progress bar only on main process
        if is_main_process(rank):
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{exp_config.num_epochs} - Training")
        else:
            train_iterator = train_loader

        for batch in train_iterator:
            optimizer.zero_grad()

            # Move batch data to device
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
                ).to(device)

                # Generate hidden states
                tc_output = teacher_model(
                    reason_inputs.input_ids,    # Input IDs
                    attention_mask=reason_inputs.attention_mask,  # Attention mask
                    output_hidden_states=True  # Get hidden states
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

            # Generate contemplation tokens from student model
            query_condensed_reasoning = f"Question: {query[0]}\nAnswer: {condensed_reasoning[0]}"
            contemp_inputs = contemp_generator.module.tokenizer(  # Note the .module access for DDP
                query_condensed_reasoning,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=exp_config.max_seq_length
            ).to(device)

            # Generate hidden states - use module directly to avoid DDP wrapping issues with tokenizer
            contemp_states = contemp_generator(
                contemp_inputs.input_ids,
                attention_mask=contemp_inputs.attention_mask
            )[:, -min(exp_config.max_contemp_tokens,contemp_inputs.input_ids.size(1)):, :]  # Get last N tokens

            # Get contemplation embeddings using sentence transformer
            if variation == 'vanilla':
                contemp_embeddings = sentence_transformer(contemp_states)

                # Reasoning loss (1 - similarity)
                similarity = sentence_transformer.compute_similarity(
                    gt_reason_embeddings, contemp_embeddings
                )

                l_reason = 1 - similarity.mean()
            elif variation == 'no_l_reason':
                pass
            elif variation == 'no_sentence_transformer':
                # cosine similarity of the hidden states
                similarity = torch.nn.functional.cosine_similarity(torch.mean(contemp_states, 1).squeeze(),
                                                                   torch.mean(gt_reason_hidden_states, 1).squeeze(), dim=-1)
                l_reason = 1 - similarity

            # Implement answer loss with teacher forcing
            # Create combined input: [query + contemplation tokens]
            combined_input = f"Question: {query[0]}\nAnswer:"
            combined_inputs = teacher_tokenizer(
                combined_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=exp_config.max_seq_length
            ).to(device)

            # Get ground truth answer tokens
            answer_inputs = teacher_tokenizer(
                ground_truth_answer,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=exp_config.max_seq_length
            ).to(device)

            # Compute loss using the modified function with mode="train"
            l_ans = compute_loss_ans(contemp_states, teacher_model, teacher_tokenizer,
                               answer_loss_fn, combined_inputs, answer_inputs,
                               exp_config, device, mode="train")

            # Combined loss
            if variation == 'no_l_reason':
                loss = l_ans
            else:
                loss = exp_config.alpha * l_reason + (1 - exp_config.alpha) * l_ans

            # Backpropagate
            loss.backward()
            optimizer.step()

            # Track losses locally
            total_loss += loss.item()
            if variation != 'no_l_reason':
                reason_loss += l_reason.item()
            ans_loss += l_ans.item()

        # Synchronize losses across processes
        loss_tensor = torch.tensor([total_loss, reason_loss, ans_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        # Calculate average losses
        total_loss = loss_tensor[0].item() / (len(train_loader) * world_size)
        reason_loss = loss_tensor[1].item() / (len(train_loader) * world_size)
        ans_loss = loss_tensor[2].item() / (len(train_loader) * world_size)

        # Log metrics (only on main process)
        if is_main_process(rank):
            logger.log_metrics({
                "total_loss": total_loss,
                "reason_loss": reason_loss,
                "ans_loss": ans_loss
            }, epoch)

            print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} (Reason: {reason_loss:.4f}, Ans: {ans_loss:.4f})")

        # Evaluate on validation set
        if eval_dataset:
            eval_loss = evaluate_distributed(
                contemp_generator,
                sentence_transformer,
                eval_loader,
                model_config,
                exp_config,
                variation,
                rank,
                world_size
            )

            # Log validation metrics (only on main process)
            if is_main_process(rank):
                logger.log_metrics({"eval_loss": eval_loss}, epoch)
                print(f"Validation Loss: {eval_loss:.4f}")

                # Save checkpoint (only on main process)
                if eval_loss < best_val_loss:
                    best_val_loss = eval_loss
                    ckpt_path = f"{exp_config.checkpoint_path}/contemp_generator_best"
                    utils.create_directory(ckpt_path)
                    contemp_generator.module.save_pretrained(ckpt_path)  # Save unwrapped model
                    print(f"Saved best model with validation loss: {best_val_loss:.4f}")

        # Synchronize best_val_loss across processes
        best_val_loss_tensor = torch.tensor([best_val_loss], device=device)
        dist.all_reduce(best_val_loss_tensor, op=dist.ReduceOp.MIN)
        best_val_loss = best_val_loss_tensor.item()

        # Save periodic checkpoints (only on main process)
        if is_main_process(rank) and (epoch + 1) % exp_config.save_interval == 0:
            ckpt_path = f"{exp_config.checkpoint_path}/contemp_generator_epoch{epoch+1}"
            utils.create_directory(ckpt_path)
            contemp_generator.module.save_pretrained(ckpt_path)  # Save unwrapped model

    # Final cleanup and return (only on main process)
    if is_main_process(rank):
        # Save final model
        model_path = f"{exp_config.model_save_path}/contemp_generator"
        utils.create_directory(model_path)
        contemp_generator.module.save_pretrained(model_path)  # Save unwrapped model
        logger.close()

    # Make sure all processes wait for the main process to save
    dist.barrier()

    return contemp_generator.module  # Return unwrapped model


def evaluate_distributed(contemp_generator, sentence_transformer, eval_loader, model_config, exp_config, variation, rank, world_size):
    device = rank

    # Ensure the contemp_generator is in evaluation mode during evaluation
    contemp_generator.eval()

    # Initialize teacher model and tokenizer for evaluation
    teacher_model = LlamaForCausalLM.from_pretrained(model_config.teacher_model_name)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    teacher_tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Initialize answer loss function
    answer_loss_fn = nn.CrossEntropyLoss(ignore_index=teacher_tokenizer.pad_token_id)

    total_loss = 0
    reason_loss_sum = 0
    ans_loss_sum = 0

    # Create progress bar only on main process
    if is_main_process(rank):
        eval_iterator = tqdm(eval_loader, desc="Evaluating")
    else:
        eval_iterator = eval_loader

    with torch.no_grad():  # No gradients for evaluation
        for batch in eval_iterator:
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

            # Generate contemplation tokens from student model - use .module to access tokenizer
            query_inputs = contemp_generator.module.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=exp_config.max_seq_length
            ).to(device)

            # Generate hidden states
            contemp_states = contemp_generator(
                query_inputs.input_ids,
                attention_mask=query_inputs.attention_mask
            )

            if variation == 'vanilla':
                # Get contemplation embeddings
                contemp_embeddings = sentence_transformer(
                    contemp_states
                )

                # Get reasoning embeddings
                gt_reason_embeddings = sentence_transformer(
                gt_reason_hidden_states
                )

                # Compute similarity for reasoning loss
                similarity = sentence_transformer.compute_similarity(
                    gt_reason_embeddings, contemp_embeddings
                )
                # Reasoning loss (1 - similarity)
                l_reason = 1 - similarity.mean()

            elif variation == 'no_sentence_transformer':
                similarity = torch.nn.functional.cosine_similarity(torch.mean(contemp_states, 1).squeeze(),
                                                                   torch.mean(gt_reason_hidden_states, 1).squeeze(), dim=-1)
                l_reason = 1 - similarity
            elif variation == 'no_l_reason':
                pass

            # Implement answer loss - using the same teacher forcing approach
            # Create combined input: [query + contemplation tokens]
            combined_input = f"Question: {query[0]}\nAnswer:"
            combined_inputs = teacher_tokenizer(
                combined_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=exp_config.max_seq_length
            ).to(device)

            # Get ground truth answer tokens
            answer_inputs = teacher_tokenizer(
                ground_truth_answer,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=exp_config.max_seq_length
            ).to(device)

            # Compute answer loss with mode="eval"
            l_ans = compute_loss_ans(contemp_states, teacher_model, teacher_tokenizer,
                               answer_loss_fn, combined_inputs, answer_inputs,
                               exp_config, device, mode="eval")

            # Combined loss with alpha weighting
            if variation == 'no_l_reason':
                loss = l_ans
            else:
                loss = exp_config.alpha * l_reason + (1 - exp_config.alpha) * l_ans

            # Track losses
            total_loss += loss.item()
            if variation != 'no_l_reason':
                reason_loss_sum += l_reason.item()
            ans_loss_sum += l_ans.item()

    # Synchronize losses across processes
    loss_tensor = torch.tensor([total_loss, reason_loss_sum, ans_loss_sum], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

    # Calculate averages
    total_loss = loss_tensor[0].item() / (len(eval_loader) * world_size)
    reason_loss = loss_tensor[1].item() / (len(eval_loader) * world_size)
    ans_loss = loss_tensor[2].item() / (len(eval_loader) * world_size)

    # Print detailed evaluation metrics (only on main process)
    if is_main_process(rank):
        print(f"Evaluation - Total Loss: {total_loss:.4f}, Reason Loss: {reason_loss:.4f}, Answer Loss: {ans_loss:.4f}")

    return total_loss