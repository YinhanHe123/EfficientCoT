import copy
import torch
import torch.optim as optim
from tqdm import tqdm
import os
from models.codi_model import CODIModel
from utils.logging import Logger

def train_codi_model(
    base_model_name,
    train_dataset,
    eval_dataset,
    output_path,
    num_continuous_tokens=6,
    learning_rate=8e-4,
    weight_decay=0.01,
    num_epochs=1,
    alpha=1.0,
    beta=1.0,
    gamma=20.0,
    device="cuda",
    eval_steps=25
):
    """
    Train a CODI model using the self-distillation framework

    Args:
        base_model_name: Name of the base model
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        output_path: Path to save the model
        num_continuous_tokens: Number of continuous thought tokens
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        alpha: Weight for teacher loss
        beta: Weight for student loss
        gamma: Weight for knowledge distillation loss
        device: Device to run the model on
        eval_steps: Number of steps between evaluations

    Returns:
        Trained CODI model
    """
    # ------------------BEGIN DEBUGGING-----------------
    def check_embeddings_update(model, tokenizer, tokens=["<bot>", "<eot>"]):
        with torch.no_grad():
            embeddings = model.get_input_embeddings()
            for token in tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                norm = torch.norm(embeddings.weight[token_id]).item()
                print(f"Token {token} (ID: {token_id}) embedding norm: {norm:.4f}")
    # ------------------END DEBUGGING-----------------
    # Initialize the CODI model
    codi_model = CODIModel(
        base_model_name=base_model_name,
        num_continuous_tokens=num_continuous_tokens,
        device=device
    )

    # Apply LoRA for efficient fine-tuning
    codi_model = codi_model.apply_lora()
    codi_model.model.model.lm_head.weight.requires_grad = True
    codi_model.model.model.model.embed_tokens.weight.requires_grad=True

    def freeze_old_weights_hook(grad):
        return torch.nan_to_num(grad, nan=0, posinf=0, neginf=0) * torch.concat([torch.zeros_like(grad[:-1]), torch.ones_like(grad[-1:])], dim=0).to(grad.device)

    lm_head_hooks = codi_model.model.model.lm_head.weight.register_hook(freeze_old_weights_hook)
    embed_tokens_hooks = codi_model.model.model.model.embed_tokens.weight.register_hook(freeze_old_weights_hook)

    # Create output directory
    os.makedirs(f"{output_path}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_path}/logs", exist_ok=True)

    # Setup optimizer
    optimizer = optim.AdamW(
        [p for p in codi_model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )

#     optimizer = optim.AdamW(
#     [
#         {'params': [p for n, p in codi_model.named_parameters() if p.requires_grad and 'embeddings' in n], 'lr': learning_rate * 2},  # Higher learning rate for embeddings
#         {'params': [p for n, p in codi_model.named_parameters() if p.requires_grad and 'embeddings' not in n], 'lr': learning_rate}
#     ],
#     weight_decay=weight_decay
# )

    # Setup logger
    logger = Logger(
        log_dir=f"{output_path}/logs",
        experiment_name="codi_training"
    )

    # Training loop
    best_val_loss, global_step = float('inf'), 0
    # -----------------BEGIN DEBUGGING-----------------
    print("Initial token embedding norms:")
    check_embeddings_update(codi_model.model, codi_model.tokenizer)
    # -----------------END DEBUGGING-----------------
    for epoch in range(num_epochs):
        codi_model.train()
        total_loss, teacher_loss, student_loss, kd_loss  = 0, 0, 0, 0

        # ------------------BEGIN DEBUGGING-----------------
        print(f"\nToken embedding norms before epoch {epoch+1}:")
        check_embeddings_update(codi_model.model, codi_model.tokenizer)
        # ------------------END DEBUGGING-----------------
        for sample in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            sample_t_loss, sample_s_loss, sample_kd_loss = codi_model(sample)
            sample_loss = alpha * sample_t_loss + beta * sample_s_loss + gamma * sample_kd_loss
            sample_loss.backward()
            torch.nn.utils.clip_grad_norm_(codi_model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track losses
            total_loss += sample_loss.item()
            teacher_loss += sample_t_loss.item()
            student_loss += sample_s_loss.item()
            kd_loss += sample_kd_loss.item()
            global_step += 1

            # Log every eval_steps
            if global_step % eval_steps == 0:
                # -----------------BEGIN DEBUGGING-----------------
                print(f"\nToken embedding norms at step {global_step}:")
                check_embeddings_update(codi_model.model, codi_model.tokenizer)
                # -----------------END DEBUGGING-----------------
                logger.log_metrics({
                    "train/loss": sample_loss.item(),
                    "train/teacher_loss": sample_t_loss.item(),
                    "train/student_loss": sample_s_loss.item(),
                    "train/kd_loss": sample_kd_loss.item()
                }, global_step)

                print(f"Step {global_step} - Loss: {sample_loss.item():.4f} "
                      f"(Teacher: {sample_t_loss.item():.4f}, "
                      f"Student: {sample_s_loss.item():.4f}, "
                      f"KD: {sample_kd_loss.item():.4f})")
            del sample_t_loss, sample_s_loss, sample_kd_loss, sample_loss
            torch.cuda.empty_cache()

        # Calculate average losses for the epoch
        avg_loss = total_loss / len(train_dataset)
        avg_teacher_loss = teacher_loss / len(train_dataset)
        avg_student_loss = student_loss / len(train_dataset)
        avg_kd_loss = kd_loss / len(train_dataset)

        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f} "
              f"(Teacher: {avg_teacher_loss:.4f}, Student: {avg_student_loss:.4f}, KD: {avg_kd_loss:.4f})")

        # Evaluate on validation set
        eval_teacher_loss, eval_student_loss, eval_kd_loss = evaluate_codi(codi_model,eval_dataset)
        eval_loss = (alpha * eval_teacher_loss + beta * eval_student_loss + gamma * eval_kd_loss).item()
        logger.log_metrics({
            "val/loss": eval_loss,
            "val/teacher_loss": eval_teacher_loss.item(),
            "val/student_loss": eval_student_loss.item(),
            "val/kd_loss": eval_kd_loss.item()
        }, global_step)

        print(f"Validation Loss: {eval_loss:.4f} "
              f"(Teacher: {eval_teacher_loss.item():.4f}, "
              f"Student: {eval_student_loss.item():.4f}, "
              f"KD: {eval_kd_loss.item():.4f})")

        # Save the best model
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss

            # Check if model has LoRA adapter
            if hasattr(codi_model.model, 'merge_and_unload'):
                # Merge and save the model
                # merged_model = copy.deepcopy(codi_model)
                # merged_model.model = merged_model.model.merge_and_unload()
                # merged_model.save_pretrained(output_path)
                codi_model.model = codi_model.model.merge_and_unload()
                codi_model.save_pretrained(output_path)
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
                torch.cuda.empty_cache()
                # del merged_model
                # torch.cuda.empty_cache()
            else:
                # Save the model directly
                codi_model.save_pretrained(output_path)
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        # Clean up memory
        torch.cuda.empty_cache()
        # ------------------BEGIN DEBUGGING-----------------
        print(f"\nToken embedding norms after epoch {epoch+1}:")
        check_embeddings_update(codi_model.model, codi_model.tokenizer)
        # ------------------END DEBUGGING-----------------
    del codi_model
    torch.cuda.empty_cache()
    logger.close()
    lm_head_hooks.remove()
    embed_tokens_hooks.remove()

def evaluate_codi(codi_model, eval_dataset):
    """
    Evaluate the CODI model on the validation set

    Args:
        codi_model: The CODI model to evaluate
        eval_dataset: Evaluation dataset
    Returns:
        Average teacher loss, student loss, and kd loss
    """
    codi_model.eval()
    teacher_loss, student_loss, kd_loss  = 0, 0, 0
    with torch.no_grad():
        for sample in tqdm(eval_dataset, desc="Evaluating"):
            # Process batch data
            sample_t_loss, sample_s_loss, sample_kd_loss = codi_model(sample)
            teacher_loss += sample_t_loss.cpu() / len(eval_dataset)
            student_loss += sample_s_loss.cpu() / len(eval_dataset)
            kd_loss += sample_kd_loss.cpu() / len(eval_dataset)
            del sample_t_loss, sample_s_loss, sample_kd_loss
    return teacher_loss, student_loss, kd_loss