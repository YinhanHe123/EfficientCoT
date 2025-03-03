import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM
import utils.utils as utils
from utils.logging import Logger

def compute_loss_ans(contemp_states, teacher_model, teacher_tokenizer, answer_loss_fn, combined_inputs, answer_inputs, exp_config, device, mode="train"):
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
# def compute_loss_ans(contemp_states, teacher_model, teacher_tokenizer, answer_loss_fn, combined_inputs, answer_inputs, exp_config, device, mode="train"):
#     # Determine whether to compute gradients based on mode
#     # In 'train' mode, we'll allow gradients to flow back to contemp_states
#     # In 'eval' mode, we'll ensure no gradients are computed
#     context_manager = torch.enable_grad() if mode == "train" else torch.no_grad()

#     with context_manager:
#         # Get the total sequence length and limit contemp states to max_contemp_tokens
#         contemp_len = min(contemp_states.size(1), exp_config.max_contemp_tokens)

#         # With torch.no_grad for teacher model operations
#         with torch.no_grad():
#             # Get the embeddings from the model's embedding layer
#             inputs_embeds = teacher_model.get_input_embeddings()(combined_inputs.input_ids)
#             answer_embeds = teacher_model.get_input_embeddings()(answer_inputs.input_ids)

#             # Create a new inputs_embeds by concatenating with contemp_states
#             # In train mode, we need to preserve the computation graph
#             # In eval mode, we can detach to save memory
#             contemp_states_to_use = contemp_states if mode == "train" else contemp_states.detach()

#             combined_embeds = torch.cat([
#                 inputs_embeds,
#                 contemp_states_to_use[:, -contemp_len:, :],
#                 answer_embeds
#             ], dim=1)

#             # Create a proper attention mask that covers both parts
#             attention_mask = torch.ones(
#                 (combined_inputs.input_ids.size(0), combined_embeds.shape[1]),
#                 dtype=torch.long,
#                 device=device
#             )

#             # Create position ids that account for both parts
#             position_ids = torch.arange(
#                 combined_embeds.shape[1],
#                 dtype=torch.long,
#                 device=device
#             ).unsqueeze(0).expand(combined_inputs.input_ids.size(0), -1)

#             # Forward pass with combined embeddings
#             teacher_outputs = teacher_model(
#                 inputs_embeds=combined_embeds,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 output_hidden_states=True
#             )

#             # Get logits from the teacher model
#             logits = teacher_outputs.logits

#         # Get answer labels (shifted by one token)
#         answer_labels = answer_inputs.input_ids[:, 1:]  # Shifted by one token

#         # Get the index to start predictions from (where the answer begins)
#         # This is where the original input ends plus the contemplation tokens
#         start_idx = combined_inputs.input_ids.size(1) + contemp_len - 1
#         seq_length = answer_labels.size(1)

#         # Get all relevant logits at once
#         answer_logits = logits[:, start_idx:start_idx+seq_length, :]
#         # Reshape for the loss function
#         answer_logits_flat = answer_logits.reshape(-1, answer_logits.size(-1))
#         answer_labels_flat = answer_labels.reshape(-1)

#         # Calculate loss for all positions at once
#         l_ans = answer_loss_fn(answer_logits_flat, answer_labels_flat)

#     return l_ans


def train_contemplation_generator(
    contemp_generator,
    sentence_transformer,
    train_dataset,
    eval_dataset,
    model_config,
    exp_config,
    variation
):
    device = exp_config.device
    contemp_generator = contemp_generator.to(device)

    # Ensure sentence transformer is in evaluation mode
    if variation == 'vanilla':
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

    # Setup logger
    logger = Logger(
        log_dir=exp_config.log_dir,
        experiment_name=f"contemp_generator"
    )
    logger.log_hyperparams(exp_config.__dict__ | model_config.__dict__)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(exp_config.num_epochs):
        # Set contemp_generator to train mode only during training
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

            # Track losses
            total_loss += loss.item()
            if variation != 'no_l_reason':
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
        if (epoch + 1) % 5 == 0:
            eval_loss = evaluate(
                contemp_generator,
                sentence_transformer,
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
                best_state_dict = contemp_generator.state_dict()

        # # Save checkpoint
        # if (epoch + 1) % exp_config.save_interval == 0:
        #     ckpt_path = f"{exp_config.checkpoint_path}/contemp_generator_epoch{epoch+1}"
        #     utils.create_directory(ckpt_path)
        #     contemp_generator.save_pretrained(ckpt_path)
    # save best contemp_generator
    contemp_generator.load_state_dict(best_state_dict)
    model_path = f"{exp_config.model_save_path}/contemp_generator"
    utils.create_directory(model_path)
    contemp_generator.save_pretrained(model_path)
    print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    logger.close()
    return contemp_generator


def evaluate(contemp_generator, sentence_transformer, eval_dataset, model_config, exp_config, variation='vanilla'):
    device = contemp_generator.device

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

    with torch.no_grad():  # No gradients for evaluation
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

    # Calculate averages
    avg_total_loss = total_loss / len(eval_dataset)
    avg_reason_loss = reason_loss_sum / len(eval_dataset)
    avg_ans_loss = ans_loss_sum / len(eval_dataset)

    # Print detailed evaluation metrics
    print(f"Evaluation - Total Loss: {avg_total_loss:.4f}, Reason Loss: {avg_reason_loss:.4f}, Answer Loss: {avg_ans_loss:.4f}")

    return avg_total_loss