import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import utils.utils as utils
from utils.logging import Logger

def compute_loss_ans(contemp_states, teacher_model, teacher_tokenizer, answer_loss_fn, combined_input_for_query, combined_input_for_answer, answer_inputs, exp_config, device, mode="train"):
    # Determine whether to compute gradients based on mode
    context_manager = torch.enable_grad() if mode == "train" else torch.no_grad()

    with context_manager:
        # Get the total sequence length and limit contemp states to train_max_contemp_tokens
        contemp_len = min(contemp_states.size(1), exp_config.train_max_contemp_tokens)

        # Get the embeddings from the model's embedding layer (no gradients needed)
        with torch.no_grad():
            inputs_embeds_for_query = teacher_model.get_input_embeddings()(combined_input_for_query.input_ids)
            inputs_embeds_for_answer = teacher_model.get_input_embeddings()(combined_input_for_answer.input_ids)
            underscore_tokens = (teacher_tokenizer.eos_token+' ')*(answer_inputs.input_ids.shape[1])
            underscore_tokens = underscore_tokens.strip()
            answer_underscore = teacher_model.get_input_embeddings()(teacher_tokenizer(underscore_tokens, return_tensors="pt", add_special_tokens=False)['input_ids'].to(device))

        # Keep contemp_states gradients in train mode
        contemp_states_to_use = contemp_states if mode == "train" else contemp_states.detach()

        # Create a new inputs_embeds by concatenating
        combined_embeds = torch.cat([
            inputs_embeds_for_query,
            contemp_states_to_use,
            inputs_embeds_for_answer,
            answer_underscore
        ], dim=1)
       
        attention_mask = torch.ones(
            (combined_embeds.shape[0], combined_embeds.shape[1]),
            dtype=torch.long,
            device=device
        )
        position_ids = torch.arange(
            combined_embeds.shape[1],
            dtype=torch.long,
            device=device
        ).unsqueeze(0).expand(combined_input_for_query.input_ids.size(0), -1)

        # Forward pass with combined embeddings - conditionally use no_grad
        teacher_outputs = teacher_model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True
        )

        # Get logits from the teacher model
        logits = teacher_outputs.logits

        # Get answer labels (shifted by one token)
        answer_labels_flat = answer_inputs.input_ids.reshape(-1)

        # Get all relevant logits at once
        answer_logits = logits[:, -(answer_inputs.input_ids.shape[1]+1):-1, :]
        # Reshape for the loss function
        answer_logits_flat = answer_logits.reshape(-1, answer_logits.size(-1))

        # Calculate loss for all positions at once
        l_ans = answer_loss_fn(answer_logits_flat, answer_labels_flat)
    return l_ans

def format_combined_input(query):
    """
    Format combined input based on the model type
    """
    combined_input_for_query = f"[INST] Question: {query}"
    combined_input_for_answer = "Answer: "
    return (combined_input_for_query, combined_input_for_answer)

def train_contemplation_generator(
    contemp_generator,
    sentence_transformer,
    train_dataset,
    eval_dataset,
    model_config,
    exp_config,
    variation
):
    # Setup logger
    logger = Logger(
        log_dir=exp_config.log_dir,
        experiment_name=exp_config.experiment_name
    )
    logger.logger.info("Training contemplation generator")
    
    device = exp_config.device
    contemp_generator = contemp_generator.to(device)
    
    # Initialize teacher model - use AutoModelForCausalLM to handle different model types
    teacher_model = AutoModelForCausalLM.from_pretrained(model_config.teacher_model_name)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Set to evaluation mode
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Initialize teacher tokenizer for answer generation
    teacher_tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Ensure sentence transformer is in evaluation mode
    if variation == 'vanilla':
        sentence_transformer = sentence_transformer.to(device)
        sentence_transformer.eval()
        for param in sentence_transformer.parameters():
            param.requires_grad = False
        
    for data in [train_dataset, eval_dataset]:
        for idx in tqdm(range(len(data))):
            if 'gt_reason_hidden' not in data[idx]:
                original_inputs = teacher_tokenizer(
                    data[idx]["reasoning"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=exp_config.max_seq_length
                ).to(device)
                with torch.no_grad():
                    original_outputs = teacher_model(
                        **original_inputs,
                        output_hidden_states=True
                    )
                gt_reason_hidden = original_outputs.hidden_states[exp_config.start_layer_idx]
            else:
                gt_reason_hidden = data[idx]['gt_reason_hidden'].to(device)
            if variation == "vanilla":
                gt_reason_hidden = sentence_transformer(gt_reason_hidden)
            data.update_item(idx, "gt_reason_hidden", gt_reason_hidden.cpu())
            del gt_reason_hidden
            torch.cuda.empty_cache()
        
    # Initialize answer loss function
    answer_loss_fn = nn.CrossEntropyLoss(ignore_index=teacher_tokenizer.pad_token_id)

    # Training loop
    print("Starting training contemplation generator...")
    for param in contemp_generator.student_model.parameters():
        param.requires_grad = False
    for (lr, wd, ne) in [(exp_config.cg_linear_lr, exp_config.cg_linear_wd, exp_config.cg_linear_epochs), (exp_config.cg_llm_lr, exp_config.cg_llm_wd, exp_config.cg_llm_epochs)]:
        best_val_loss = float('inf')
        optimizer = optim.AdamW(contemp_generator.parameters(), lr=lr, weight_decay=wd)
        for epoch in (range(ne)):
            contemp_generator.train()
            total_loss, reason_loss, ans_loss = 0, 0, 0
            for idx in tqdm(random.sample(range(len(train_dataset)), len(train_dataset)), desc=f"Epoch {epoch+1}/{ne}"): 
                item = train_dataset[idx]
                optimizer.zero_grad()                
                loss, total_loss, reason_loss, ans_loss = process_item(
                    "train",
                    item,
                    contemp_generator,
                    teacher_model,
                    teacher_tokenizer,
                    sentence_transformer,
                    answer_loss_fn,
                    exp_config,
                    device,
                    variation,
                    total_loss,
                    reason_loss,
                    ans_loss
                )

                # Backpropagate
                loss.backward()
                optimizer.step()
                # clean cache
                torch.cuda.empty_cache()

            # Calculate average losses
            avg_total_loss = total_loss / len(train_dataset)
            avg_reason_loss = reason_loss / len(train_dataset)
            avg_ans_loss = ans_loss / len(train_dataset)

            eval_loss = evaluate(
                teacher_model,
                teacher_tokenizer,
                contemp_generator,
                sentence_transformer,
                eval_dataset,
                exp_config,
                variation
            )

            # Log metrics
            logger.log_metrics({
                "total_loss": avg_total_loss,
                "reason_loss": avg_reason_loss,
                "ans_loss": avg_ans_loss,
                "eval_loss": eval_loss
            }, epoch)
            print(f"Epoch {epoch+1}/{ne} - Train Loss: {avg_total_loss:.4f} (Reason: {avg_reason_loss:.4f}, Ans: {avg_ans_loss:.4f}), Val Loss: {eval_loss:.4f}")

            # Save best model
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                model_path = f"{exp_config.model_save_path}/contemp_generator"
                utils.create_directory(model_path)
                contemp_generator.save_pretrained(model_path)
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        contemp_generator = contemp_generator.from_pretrained(model_path).to(device)
        logger.logger.info(f"Loading best validation loss = {best_val_loss}")
        print(f"Loading best validation loss = {best_val_loss}")
        for param in contemp_generator.student_model.parameters():
            param.requires_grad = True
    logger.close()
    del teacher_model
    torch.cuda.empty_cache()
    return contemp_generator

def process_item(mode, item, contemp_generator, teacher_model, teacher_tokenizer, sentence_transformer, answer_loss_fn, exp_config, device, variation, total_loss=0, reason_loss=0, ans_loss=0):
    # Process batch
    query = item["query"]
    ground_truth_answer = item["answer"]
    gt_reason = item['gt_reason_hidden'].to(device)
    query_prompt, answer_prompt = format_combined_input(query)

    # Find position of contemplation tokens
    query_inputs = contemp_generator.tokenizer(
        query_prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        max_length=exp_config.max_seq_length
    ).to(device)
    prefix_length = query_inputs.input_ids.size(1) - 1
    
    answer_inputs = contemp_generator.tokenizer(
        answer_prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        max_length=exp_config.max_seq_length,
        add_special_tokens=False
    ).to(device)
    
    # Add contemplation tokens BEFORE the "Answer:" part
    contemp_inputs = torch.cat([
        query_inputs['input_ids'], 
        torch.tensor([[contemp_generator.tokenizer.eos_token_id * exp_config.train_max_contemp_tokens]]).to(device), 
        answer_inputs['input_ids']
    ], dim=1)

    # Generate hidden states
    all_contemp_states = contemp_generator(contemp_inputs, attention_mask=torch.ones_like(contemp_inputs))

    # Extract the contemplation tokens from the correct position (after prefix, before "Answer:")
    contemp_states = all_contemp_states[:, prefix_length:prefix_length+exp_config.train_max_contemp_tokens, :]

    # Get contemplation embeddings using sentence transformer
    if variation == 'vanilla':
        contemp_embeddings = sentence_transformer(contemp_states)

        # Reasoning loss (1 - similarity)
        similarity = sentence_transformer.compute_similarity(gt_reason, contemp_embeddings)
        l_reason = 1 - similarity.mean()
    elif variation == 'no_l_reason':
        pass
    elif variation == 'no_sentence_transformer':
        # cosine similarity of the hidden states
        similarity = torch.nn.functional.cosine_similarity(torch.mean(contemp_states, 1).squeeze(),
                                                          torch.mean(gt_reason, 1).squeeze(), dim=-1)
        l_reason = 1 - similarity

    query_inputs = teacher_tokenizer(
        query_prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        max_length=exp_config.max_seq_length
    ).to(device)
    
    answer_prompt_inputs = teacher_tokenizer(
        answer_prompt,
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
        max_length=exp_config.max_seq_length,
        add_special_tokens=False
    ).to(device)

    # Compute loss using the modified function with mode="train"
    l_ans = compute_loss_ans(contemp_states, teacher_model, teacher_tokenizer,
                       answer_loss_fn, query_inputs, answer_prompt_inputs, answer_inputs,
                       exp_config, device, mode)

    # Combined loss
    if variation == 'no_l_reason':
        loss = l_ans
    else:
        loss = exp_config.alpha * l_reason + (1 - exp_config.alpha) * l_ans

    # Track losses
    total_loss += loss.item()
    if variation != 'no_l_reason':
        reason_loss += l_reason.item()
    ans_loss += l_ans.item()

    return loss, total_loss, reason_loss, ans_loss

def evaluate(teacher_model, teacher_tokenizer, contemp_generator, sentence_transformer, eval_dataset, exp_config, variation='vanilla'):
    device = exp_config.device

    # Ensure the contemp_generator is in evaluation mode during evaluation
    contemp_generator.eval()
    teacher_model.eval()
    answer_loss_fn = nn.CrossEntropyLoss(ignore_index=teacher_tokenizer.pad_token_id)

    total_loss = 0
    reason_loss_sum = 0
    ans_loss_sum = 0

    with torch.no_grad():  # No gradients for evaluation
        for item in tqdm(eval_dataset, desc="Evaluating"):
            # Process batch  
            _, total_loss, reason_loss_sum, ans_loss_sum = process_item(
                "eval",
                item,
                contemp_generator,
                teacher_model,
                teacher_tokenizer,
                sentence_transformer,
                answer_loss_fn,
                exp_config,
                device,
                variation,
                total_loss,
                reason_loss_sum,
                ans_loss_sum
            )

    # Calculate averages
    avg_total_loss = total_loss / len(eval_dataset)
    avg_reason_loss = reason_loss_sum / len(eval_dataset)
    avg_ans_loss = ans_loss_sum / len(eval_dataset)

    # Print detailed evaluation metrics
    print(f"Evaluation - Total Loss: {avg_total_loss:.4f}, Reason Loss: {avg_reason_loss:.4f}, Answer Loss: {avg_ans_loss:.4f}")

    return avg_total_loss