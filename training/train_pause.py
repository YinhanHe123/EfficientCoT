# import os
# import torch
# from torch.utils.data import Dataset
# from transformers import Trainer, TrainingArguments


# class PauseFinetuningDataset(Dataset):
#     """Dataset for pause-finetuning (Algorithm 2 in the paper)"""

#     def __init__(self, original_dataset, tokenizer, mft=5, max_length=512):
#         self.dataset = original_dataset
#         self.tokenizer = tokenizer
#         self.mft = mft  # Number of pause tokens to append
#         self.max_length = max_length
#         self.pause_token_id = tokenizer.convert_tokens_to_ids("<pause>")

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         query = item["query"]
#         answer = item.get("answer", "")

#         # Tokenize query
#         query_tokens = self.tokenizer(query, return_tensors="pt",
#                                      truncation=True, max_length=self.max_length)
#         query_ids = query_tokens.input_ids[0]

#         # Create pause tokens
#         pause_ids = [self.pause_token_id] * self.mft

#         # Tokenize answer
#         answer_tokens = self.tokenizer(answer, return_tensors="pt",
#                                       truncation=True, max_length=self.max_length)
#         answer_ids = answer_tokens.input_ids[0]

#         # Combine query + pause tokens as input, answer as target
#         input_ids = torch.cat([query_ids, torch.tensor(pause_ids, dtype=torch.long)])

#         # Create attention mask
#         attention_mask = torch.ones(len(input_ids), dtype=torch.long)

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels":  torch.cat([input_ids, answer_ids]),
#             "query_length": len(query_ids),
#             "pause_length": self.mft
#         }

# def custom_data_collator(features):
#     """Custom collator for variable length sequences"""
#     batch = {}

#     # Pad input_ids to max length in batch
#     max_length = max(len(f["input_ids"]) for f in features)
#     batch["input_ids"] = torch.stack([
#         torch.cat([f["input_ids"],
#                   torch.full((max_length - len(f["input_ids"]),),
#                             0, dtype=torch.long)])
#         for f in features
#     ])

#     # Pad attention_mask
#     batch["attention_mask"] = torch.stack([
#         torch.cat([f["attention_mask"],
#                   torch.zeros(max_length - len(f["attention_mask"]),
#                              dtype=torch.long)])
#         for f in features
#     ])

#     # Handle labels (may be different length)
#     if "labels" in features[0]:
#         max_label_length = max(len(f["labels"]) for f in features)
#         batch["labels"] = torch.stack([
#             torch.cat([f["labels"],
#                       torch.full((max_label_length - len(f["labels"]),),
#                                -100, dtype=torch.long)])
#             for f in features
#         ])
#     return batch

# def train_pause_model(tokenizer, model, train_dataset, eval_dataset, experiment_config):
#     """
#     Train a model with pause tokens as per the paper
#     """

#     # Set pad token if not set
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # Add pause token to vocabulary
#     pause_token = "<pause>"
#     if pause_token not in tokenizer.get_vocab():
#         special_tokens = {"additional_special_tokens": [pause_token]}
#         num_added = tokenizer.add_special_tokens(special_tokens)
#         if num_added > 0:
#             model.resize_token_embeddings(len(tokenizer))

#     # Create appropriate dataset based on mode
#     train_pause_dataset = PauseFinetuningDataset(
#         train_dataset,
#         tokenizer,
#         mft=experiment_config.train_max_contemp_tokens
#     )
#     eval_pause_dataset = PauseFinetuningDataset(
#         eval_dataset,
#         tokenizer,
#         mft=experiment_config.train_max_contemp_tokens
#     )
#     # Setup training arguments
#     training_args = TrainingArguments(
#         output_dir=os.path.join(experiment_config.model_save_path, "pause"),
#         per_device_train_batch_size=4,
#         per_device_eval_batch_size=4,
#         gradient_accumulation_steps=4,
#         learning_rate=2e-5,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         warmup_steps=500,
#         logging_dir=os.path.join(experiment_config.log_dir, "pause"),
#         logging_steps=10,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         load_best_model_at_end=True,
#         fp16=True if torch.cuda.is_available() else False,
#     )

#     # Create trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_pause_dataset,
#         eval_dataset=eval_pause_dataset,
#         data_collator=custom_data_collator,
#     )
#     # Train the model
#     trainer.train()

#     # Save the model
#     save_path = os.path.join(experiment_config.model_save_path, "pause")
#     os.makedirs(save_path, exist_ok=True)
#     model.save_pretrained(save_path)
#     return model


import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType


class PauseFinetuningDataset(Dataset):
    """Dataset for pause-finetuning (Algorithm 2 in the paper)"""
    def __init__(self, original_dataset, tokenizer, mft=5, max_length=512):
        self.dataset = original_dataset
        self.tokenizer = tokenizer
        self.mft = mft  # Number of pause tokens to append
        self.max_length = max_length
        self.pause_token_id = tokenizer.convert_tokens_to_ids("<pause>")
        self.answer_len = 0


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        query = item["query"]
        answer = item.get("answer", "")

        # Tokenize query
        query_tokens = self.tokenizer(query, return_tensors="pt",
                                     truncation=True, max_length=self.max_length)
        query_ids = query_tokens.input_ids[0]

        # Create pause tokens
        pause_ids = [self.pause_token_id] * self.mft

        # Tokenize answer
        answer_tokens = self.tokenizer(answer, return_tensors="pt",
                                      truncation=True, max_length=self.max_length)
        answer_ids = answer_tokens.input_ids[0][2:]
        self.answer_len = len(answer_ids)


        # Combine query + pause tokens as input, answer as target
        input_ids = torch.cat([query_ids, torch.tensor(pause_ids, dtype=torch.long)])

        # Create attention mask
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        labels = torch.cat([input_ids, answer_ids])

        return {
            "input_ids": input_ids,
            "answer_ids": answer_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "query_length": len(query_ids),
            "pause_length": self.mft
        }

def custom_data_collator(features):
    """Custom collator for variable length sequences"""
    batch = {}

    # Pad input_ids to max length in batch
    max_length = max(len(f["input_ids"]) for f in features)
    batch["input_ids"] = torch.stack([
        torch.cat([f["input_ids"],
                  torch.full((max_length - len(f["input_ids"]),),
                            0, dtype=torch.long)])
        for f in features
    ])

    # Pad attention_mask
    batch["attention_mask"] = torch.stack([
        torch.cat([f["attention_mask"],
                  torch.zeros(max_length - len(f["attention_mask"]),
                             dtype=torch.long)])
        for f in features
    ])

    # Handle labels (may be different length)
    if "labels" in features[0]:
        max_label_length = max(len(f["labels"]) for f in features)
        batch["labels"] = torch.stack([
            torch.cat([f["labels"],
                      torch.full((max_label_length - len(f["labels"]),),
                               -100, dtype=torch.long)])
            for f in features
        ])
    return batch


class PauseFinetuningTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Simplified compute_loss function
        """
        # Get inputs and labels
        input_ids = inputs["input_ids"]
        labels = inputs.pop("labels")

        # Calculate input length and expected output length
        input_len = input_ids.size(1)

        # Create causal language modeling labels by shifting
        # -100 for the input portion to exclude it from loss calculation
        lm_labels = torch.full_like(labels, -100)
        lm_labels[:, input_len-1:-1] = labels[:, input_len:]

        # Forward pass with huggingface's built-in loss calculation
        outputs = model(input_ids=labels, labels=lm_labels)
        # import pdb
        # pdb.set_trace()
        # Get the loss calculated by the model
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def train_pause_model(tokenizer, model, train_dataset, eval_dataset, experiment_config, save_path):
    """
    Train a model with pause tokens as per the paper using LoRA
    """
    # Define LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        inference_mode=False,
    )

    # Create LoRA model
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    lora_model = get_peft_model(model, lora_config)
    lora_model.train()
    lora_model.config.use_cache = False  # Disable KV cache for training with gradient checkpointing

    lora_model.model.lm_head.weight.requires_grad = True
    lora_model.model.model.embed_tokens.weight.requires_grad=True
    
    def freeze_old_weights_hook(grad):
        # import pdb
        # pdb.set_trace()
        return torch.nan_to_num(grad, nan=0, posinf=0, neginf=0) * torch.concat([torch.zeros_like(grad[:-1]), torch.ones_like(grad[-1:])], dim=0).to(grad.device)
    
    lm_head_hooks = lora_model.model.lm_head.weight.register_hook(freeze_old_weights_hook)
    embed_tokens_hooks = lora_model.model.model.embed_tokens.weight.register_hook(freeze_old_weights_hook)

    # Log trainable vs. all parameters
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Percentage of parameters being trained: {100 * trainable_params / total_params:.2f}%")

    # Create datasets
    train_pause_dataset = PauseFinetuningDataset(
        train_dataset,
        tokenizer,
        mft=experiment_config.train_max_contemp_tokens
    )

    eval_pause_dataset = PauseFinetuningDataset(
        eval_dataset,
        tokenizer,
        mft=experiment_config.train_max_contemp_tokens
    )

    # Setup training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=5e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=save_path,
        logging_steps=10,
        save_strategy="best",
        load_best_model_at_end=True,
        fp16=True if torch.cuda.is_available() else False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        # Add these to address parallel processing issues:
        dataloader_drop_last=True,  # Drop odd-sized batches
        dataloader_num_workers=0,   # No parallel data loading
        ddp_find_unused_parameters=False,
    )

    # Create custom trainer with our loss function
    trainer = PauseFinetuningTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_pause_dataset,
        eval_dataset=eval_pause_dataset,
        data_collator=custom_data_collator,
    )

    # Train the model
    trainer.train()
    lm_head_hooks.remove()
    embed_tokens_hooks.remove()
    return lora_model