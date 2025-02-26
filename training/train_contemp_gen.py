import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModel
import utils.utils as utils
from utils.logging import Logger

def train_contemplation_generator(
    contemp_generator,
    sentence_transformer,
    train_dataset,
    eval_dataset,
    config
):
    device = utils.get_device()
    contemp_generator = contemp_generator.to(device)

    # Ensure sentence transformer is in evaluation mode
    sentence_transformer = sentence_transformer.to(device)
    sentence_transformer.eval()

    # Initialize teacher model
    teacher_model = AutoModel.from_pretrained(config.teacher_model_name)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Set to evaluation mode

    # Define optimizers
    optimizer = optim.AdamW(
        contemp_generator.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Setup logger
    logger = Logger(
        log_dir=config.log_dir,
        experiment_name=f"contemp_generator"
    )
    logger.log_hyperparams(config.__dict__)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        contemp_generator.train()

        total_loss = 0
        reason_loss = 0
        ans_loss = 0

        for batch in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{config.num_epochs} - Training"):
            optimizer.zero_grad()

            # Process batch
            query = batch["query"]
            ground_truth_reasoning = batch["reasoning"]

            # Get teacher model's ground-truth reasoning hidden states
            with torch.no_grad():
                # Tokenize ground truth reasoning
                reason_inputs = teacher_model.tokenizer(
                    ground_truth_reasoning,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_seq_length
                ).to(device)

                # Get reasoning embeddings from sentence transformer
                gt_reason_embeddings = sentence_transformer(
                    reason_inputs.input_ids,
                    attention_mask=reason_inputs.attention_mask
                )

            # Generate contemplation tokens from student model
            query_inputs = contemp_generator.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_seq_length
            ).to(device)

            # Generate hidden states
            contemp_states = contemp_generator(
                query_inputs.input_ids,
                attention_mask=query_inputs.attention_mask
            )

            # Get contemplation embeddings using sentence transformer
            # Use a wrapper method to process hidden states directly
            contemp_embeddings = sentence_transformer.embed_hidden_states(
                contemp_states,
                query_inputs.attention_mask
            )

            # Reasoning loss (1 - similarity)
            with torch.no_grad():
                similarity = sentence_transformer.compute_similarity(
                    gt_reason_embeddings, contemp_embeddings
                )

            l_reason = 1 - similarity.mean()

            # TODO: Implement answer loss (Lans) based on using contemplation tokens
            # with the teacher model to generate answers
            l_ans = torch.tensor(0.0, device=device)  # Placeholder

            # Combined loss
            loss = config.alpha * l_reason + (1 - config.alpha) * l_ans

            # Backpropagate
            loss.backward()
            optimizer.step()

            # Track losses
            total_loss += loss.item()
            reason_loss += l_reason.item()
            ans_loss += l_ans.item()

        # Calculate average losses
        avg_total_loss = total_loss / len(train_dataset)
        avg_reason_loss = reason_loss / len(train_dataset)
        avg_ans_loss = ans_loss / len(train_dataset)

        # Log metrics
        logger.log_metrics({
            "total_loss": avg_total_loss,
            "reason_loss": avg_reason_loss,
            "ans_loss": avg_ans_loss
        }, epoch)

        print(f"Epoch {epoch+1} - Loss: {avg_total_loss:.4f} (Reason: {avg_reason_loss:.4f}, Ans: {avg_ans_loss:.4f})")

        # Evaluate on validation set
        if eval_dataset:
            eval_loss = evaluate(
                contemp_generator,
                sentence_transformer,
                eval_dataset,
                config
            )
            logger.log_metrics({"eval_loss": eval_loss}, epoch)
            print(f"Validation Loss: {eval_loss:.4f}")

            # Save best model
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                model_path = f"{config.model_save_path}/contemp_generator"
                utils.create_directory(model_path)
                contemp_generator.save_pretrained(model_path)
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            ckpt_path = f"{config.checkpoint_path}/contemp_generator_epoch{epoch+1}"
            utils.create_directory(ckpt_path)
            contemp_generator.save_pretrained(ckpt_path)

    logger.close()
    return contemp_generator

def evaluate(contemp_generator, sentence_transformer, eval_dataset, config):
    device = contemp_generator.device
    contemp_generator.eval()

    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataset, desc="Evaluating"):
            # Process batch
            query = batch["query"]
            ground_truth_reasoning = batch["reasoning"]

            # Tokenize ground truth reasoning
            reason_inputs = sentence_transformer.tokenizer(
                ground_truth_reasoning,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_seq_length
            ).to(device)

            # Get reasoning embeddings
            gt_reason_embeddings = sentence_transformer(
                reason_inputs.input_ids,
                attention_mask=reason_inputs.attention_mask
            )

            # Generate contemplation tokens from student model
            query_inputs = contemp_generator.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_seq_length
            ).to(device)

            # Generate hidden states
            contemp_states = contemp_generator(
                query_inputs.input_ids,
                attention_mask=query_inputs.attention_mask
            )

            # Get contemplation embeddings
            contemp_embeddings = sentence_transformer.embed_hidden_states(
                contemp_states,
                query_inputs.attention_mask
            )

            # Compute similarity
            similarity = sentence_transformer.compute_similarity(
                gt_reason_embeddings, contemp_embeddings
            )

            # Loss (1 - similarity)
            loss = 1 - similarity.mean()

            total_loss += loss.item()

    return total_loss / len(eval_dataset)