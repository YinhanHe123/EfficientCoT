import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import utils.utils as utils
from utils.logging import Logger


def compute_loss_ans(contemp_states, teacher_model, teacher_tokenizer, answer_loss_fn, combined_inputs, answer_inputs, exp_config, device, mode="train"):
# Forward with contemplation states
    # We need to provide the contemplation states to the teacher model
    # This requires modifying the hidden states of the teacher model
    with torch.no_grad():
        # Forward pass with initial tokens
        teacher_outputs = teacher_model(
            combined_inputs.input_ids,
            attention_mask=combined_inputs.attention_mask,
            output_hidden_states=True
        )

        # Get the hidden state at the injection point (start_layer_idx)
        insert_hidden_state = teacher_outputs.hidden_states[exp_config.start_layer_idx]

        # Inject contemplation tokens by replacing or concatenating
        # Here we'll use a simple approach: replace the end of the sequence with contemplation states
        # In a more sophisticated implementation, you might want to use attention to combine them
        seq_len = insert_hidden_state.size(1)
        contemp_len = min(contemp_states.size(1), exp_config.max_contemp_tokens)

        # Create a position for the contemplation tokens (after the query)
        modified_hidden_state = insert_hidden_state.clone()
        modified_hidden_state[:, -contemp_len:, :] = contemp_states[:, :contemp_len, :]

    # Forward the teacher model from this point to get logits for answer generation
    # Process through the remaining transformer layers
    current_hidden_state = modified_hidden_state

    # Get position IDs and attention mask for the forward pass
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(combined_inputs.input_ids)

    # Create position embeddings for the remaining layers if needed
    position_embeddings = None
    if hasattr(teacher_model, 'rotary_emb'):
        position_embeddings = teacher_model.rotary_emb(current_hidden_state, position_ids)

    # Process through the remaining transformer layers starting after the injection point
    for layer_idx in range(exp_config.start_layer_idx + 1, len(teacher_model.layers)):
        # Get the current layer
        layer = teacher_model.layers[layer_idx]

        # Forward through the layer but don't accumulate gradients for the layer parameters
        if mode == "train":
            with torch.enable_grad():  # Enable grad for the input but not for layer parameters
                # Store the requires_grad state
                prev_grad_states = {}
                for name, param in layer.named_parameters():
                    prev_grad_states[name] = param.requires_grad
                    param.requires_grad = False

                # Process the layer
                layer_outputs = layer(
                    current_hidden_state,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    position_embeddings=position_embeddings
                )

                # Restore the requires_grad state
                for name, param in layer.named_parameters():
                    param.requires_grad = prev_grad_states[name]
        else:
            layer_outputs = layer(
                current_hidden_state,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                position_embeddings=position_embeddings
            )

        if isinstance(layer_outputs, tuple):
            current_hidden_state = layer_outputs[0]
        else:
            current_hidden_state = layer_outputs

    # Apply the final normalization layer if available
    if hasattr(teacher_model, 'norm'):
        current_hidden_state = teacher_model.norm(current_hidden_state)

    # Generate logits using the language model head
    if hasattr(teacher_model, 'lm_head'):
        logits = teacher_model.lm_head(current_hidden_state)
    else:
        # If no lm_head is available, approximate with a linear projection
        hidden_size = current_hidden_state.size(-1)
        vocab_size = teacher_tokenizer.vocab_size
        approx_lm_head = nn.Linear(hidden_size, vocab_size).to(device)
        logits = approx_lm_head(current_hidden_state)

    # Calculate cross entropy loss with answer tokens
    # Shift the target tokens for language modeling (predict next token)
    # shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = answer_inputs.input_ids[:, 1:].contiguous()
    ans_length = shifted_labels.size(1)
    shifted_logits = logits[:, -ans_length:, :].contiguous()

    # Calculate loss
    l_ans = answer_loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                        shifted_labels.view(-1))
    return l_ans


def train_contemplation_generator(
    contemp_generator,
    sentence_transformer,
    train_dataset,
    eval_dataset,
    model_config,
    exp_config
):
    device = exp_config.device
    contemp_generator = contemp_generator.to(device)

    # Ensure sentence transformer is in evaluation mode
    sentence_transformer = sentence_transformer.to(device)
    sentence_transformer.eval()
    for param in sentence_transformer.parameters():
        param.requires_grad = False

    # Initialize teacher model
    teacher_model = AutoModel.from_pretrained(model_config.teacher_model_name)
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

    # Setup logger
    logger = Logger(
        log_dir=exp_config.log_dir,
        experiment_name=f"contemp_generator"
    )
    logger.log_hyperparams(exp_config.__dict__ | model_config.__dict__)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(exp_config.num_epochs):
        contemp_generator.train()

        total_loss = 0
        reason_loss = 0
        ans_loss = 0

        for batch in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{exp_config.num_epochs} - Training"):
            optimizer.zero_grad()

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
                ).to(device)

                # Generate hidden states
                tc_output = teacher_model(
                    reason_inputs.input_ids,    # Input IDs
                    attention_mask=reason_inputs.attention_mask,  # Attention mask
                    output_hidden_states=True  # Get hidden states
                )

                # Get the required hidden states
                gt_reason_hidden_states = tc_output.hidden_states[exp_config.start_layer_idx]

                # Get reasoning embeddings from sentence transformer
                gt_reason_embeddings = sentence_transformer(
                    gt_reason_hidden_states
                )

            # Generate contemplation tokens from student model
            query_condensed_reasoning = f"Question: {query[0]}\nAnswer: {condensed_reasoning[0]}"
            contemp_inputs = contemp_generator.tokenizer(
                query_condensed_reasoning,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=exp_config.max_seq_length
            ).to(device)

            # Generate hidden states
            contemp_states = contemp_generator(
                contemp_inputs.input_ids,
                attention_mask=contemp_inputs.attention_mask
            )[:, -min(exp_config.max_contemp_tokens,contemp_inputs.input_ids.size(1)):, :]  # Get last N tokens

            # Get contemplation embeddings using sentence transformer
            contemp_embeddings = sentence_transformer(contemp_states)

            # Reasoning loss (1 - similarity)
            similarity = sentence_transformer.compute_similarity(
                gt_reason_embeddings, contemp_embeddings
            )

            l_reason = 1 - similarity.mean()

            # Implement answer loss (Lans) - Forward contemplation tokens to teacher model
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

            l_ans = compute_loss_ans(contemp_states, teacher_model, teacher_tokenizer, answer_loss_fn, combined_inputs, answer_inputs, exp_config, device)
            # Combined loss
            loss = exp_config.alpha * l_reason + (1 - exp_config.alpha) * l_ans

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
                model_config,
                exp_config
            )
            logger.log_metrics({"eval_loss": eval_loss}, epoch)
            print(f"Validation Loss: {eval_loss:.4f}")

            # Save best model
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                model_path = f"{exp_config.model_save_path}/contemp_generator"
                utils.create_directory(model_path)
                contemp_generator.save_pretrained(model_path)
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % exp_config.save_interval == 0:
            ckpt_path = f"{exp_config.checkpoint_path}/contemp_generator_epoch{epoch+1}"
            utils.create_directory(ckpt_path)
            contemp_generator.save_pretrained(ckpt_path)

    logger.close()
    return contemp_generator



def evaluate(contemp_generator, sentence_transformer, eval_dataset, model_config, exp_config):
    device = contemp_generator.device
    contemp_generator.eval()

    # Initialize teacher model and tokenizer for evaluation
    teacher_model = AutoModel.from_pretrained(model_config.teacher_model_name)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    teacher_tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Initialize answer loss function
    answer_loss_fn = nn.CrossEntropyLoss(ignore_index=teacher_tokenizer.pad_token_id)

    total_loss = 0
    reason_loss_sum = 0
    ans_loss_sum = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataset, desc="Evaluating"):
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

            # Get reasoning embeddings
            gt_reason_embeddings = sentence_transformer(
               gt_reason_hidden_states
            )

            # Generate contemplation tokens from student model
            query_inputs = contemp_generator.tokenizer(
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

            # Get contemplation embeddings
            contemp_embeddings = sentence_transformer(
                contemp_states
            )

            # Compute similarity for reasoning loss
            similarity = sentence_transformer.compute_similarity(
                gt_reason_embeddings, contemp_embeddings
            )

            # Reasoning loss (1 - similarity)
            l_reason = 1 - similarity.mean()

            # Implement answer loss - similar to training but in eval mode
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

            l_ans = compute_loss_ans(contemp_states, teacher_model, teacher_tokenizer, answer_loss_fn, combined_inputs, answer_inputs, exp_config, device, mode="eval")
            # Combined loss with alpha weighting
            loss = exp_config.alpha * l_reason + (1 - exp_config.alpha) * l_ans

            # Track losses
            total_loss += loss.item()
            reason_loss_sum += l_reason.item()
            ans_loss_sum += l_ans.item()

    # Calculate averages
    avg_total_loss = total_loss / len(eval_dataset)
    avg_reason_loss = reason_loss_sum / len(eval_dataset)
    avg_ans_loss = ans_loss_sum / len(eval_dataset)

    # Print detailed evaluation metrics
    print(f"Evaluation - Total Loss: {avg_total_loss:.4f}, Reason Loss: {avg_reason_loss:.4f}, Answer Loss: {avg_ans_loss:.4f}")

    return avg_total_loss