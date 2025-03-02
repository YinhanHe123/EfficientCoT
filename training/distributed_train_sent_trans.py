import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from models.sentence_transformer import CustomizedSentenceTransformer
from transformers import AutoModel, AutoTokenizer
from data.reasoning_pairs import ReasoningPairsGenerator
from utils.logging import Logger
import utils.utils as utils
from utils.distributed import convert_model_to_ddp, is_main_process, reduce_loss
from torch.utils.data import DataLoader, DistributedSampler
import os
import gc

def train_sentence_transformer_distributed(
    base_model_name,
    start_layer_idx,
    end_layer_idx,
    dataset,
    config,
    rank,
    world_size
):
    """
    Train a customized sentence transformer with distributed training

    Args:
        base_model_name: Name of the base model
        start_layer_idx: Start layer for extraction
        end_layer_idx: End layer for extraction
        dataset: Dataset containing reasoning pairs
        config: Training configuration
        rank: Process rank
        world_size: Total number of processes
    """
    device = rank  # Use the current GPU rank as the device

    # Initialize the full base model for getting hidden states
    base_model = AutoModel.from_pretrained(base_model_name)
    base_model = base_model.to(device)
    base_model.eval()  # Keep it in eval mode as we only use it for features

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize sentence transformer
    sentence_transformer = CustomizedSentenceTransformer(
        base_model_name,
        start_layer_idx,
        end_layer_idx,
        config.embedding_dim
    )

    # Convert to DDP model
    sentence_transformer = convert_model_to_ddp(sentence_transformer, device)

    # Define optimizer
    optimizer = optim.AdamW(
        sentence_transformer.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Define contrastive loss
    contrastive_loss = nn.CosineEmbeddingLoss(margin=0.2)

    # Setup logger (only on main process)
    if is_main_process(rank):
        logger = Logger(
            log_dir=config.log_dir,
            experiment_name=f"sentence_transformer_{start_layer_idx}_to_{end_layer_idx}"
        )
        logger.log_hyperparams(config.__dict__)

    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Use torch.utils.data.random_split with a fixed generator for reproducibility
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Create data loaders with the samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        # Set epoch for sampler to reshuffle data
        train_sampler.set_epoch(epoch)

        # Training phase
        sentence_transformer.train()
        train_loss = 0

        # Create progress bar only on main process
        if is_main_process(rank):
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Training")
        else:
            train_iterator = train_loader

        for batch in train_iterator:
            optimizer.zero_grad()

            # Get original and condensed reasoning pairs
            original_reasoning = batch["original_reasoning"]
            condensed_reasoning = batch["condensed_reasoning"]

            # Tokenize both reasonings
            original_inputs = tokenizer(
                original_reasoning,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_seq_length
            ).to(device)

            condensed_inputs = tokenizer(
                condensed_reasoning,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_seq_length
            ).to(device)

            # Get hidden states from base model
            with torch.no_grad():
                original_outputs = base_model(
                    **original_inputs,
                    output_hidden_states=True
                )

                condensed_outputs = base_model(
                    **condensed_inputs,
                    output_hidden_states=True
                )

                # Get the hidden states from the start_layer_idx
                original_hidden_states = original_outputs.hidden_states[start_layer_idx]
                condensed_hidden_states = condensed_outputs.hidden_states[start_layer_idx]

            # Generate embeddings using the sentence transformer
            original_embeddings = sentence_transformer(
                original_hidden_states,
                attention_mask=original_inputs.attention_mask
            )

            condensed_embeddings = sentence_transformer(
                condensed_hidden_states,
                attention_mask=condensed_inputs.attention_mask
            )

            # Create targets: 1 means similar, -1 means dissimilar
            # For reasoning pairs, we expect them to be similar
            targets = torch.ones(original_embeddings.size(0), device=device)

            # Compute loss
            loss = contrastive_loss(original_embeddings, condensed_embeddings, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Calculate average loss for this process
        avg_train_loss = train_loss / len(train_loader)

        # Sync train loss across all processes
        train_loss_tensor = torch.tensor([avg_train_loss], device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = train_loss_tensor.item() / world_size

        # Validation phase
        sentence_transformer.eval()
        val_loss = 0

        # Create progress bar only on main process
        if is_main_process(rank):
            val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Validation")
        else:
            val_iterator = val_loader

        with torch.no_grad():
            for batch in val_iterator:
                # Get original and condensed reasoning pairs
                original_reasoning = batch["original_reasoning"]
                condensed_reasoning = batch["condensed_reasoning"]

                # Tokenize both reasonings
                original_inputs = tokenizer(
                    original_reasoning,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_seq_length
                ).to(device)

                condensed_inputs = tokenizer(
                    condensed_reasoning,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_seq_length
                ).to(device)

                # Get hidden states from base model
                original_outputs = base_model(
                    **original_inputs,
                    output_hidden_states=True
                )

                condensed_outputs = base_model(
                    **condensed_inputs,
                    output_hidden_states=True
                )

                # Get the hidden states from the start_layer_idx
                original_hidden_states = original_outputs.hidden_states[start_layer_idx]
                condensed_hidden_states = condensed_outputs.hidden_states[start_layer_idx]

                # Generate embeddings
                original_embeddings = sentence_transformer(
                    original_hidden_states,
                    attention_mask=original_inputs.attention_mask
                )

                condensed_embeddings = sentence_transformer(
                    condensed_hidden_states,
                    attention_mask=condensed_inputs.attention_mask
                )

                # Create targets: 1 means similar
                targets = torch.ones(original_embeddings.size(0), device=device)

                # Compute loss
                loss = contrastive_loss(original_embeddings, condensed_embeddings, targets)

                val_loss += loss.item()

        # Calculate average validation loss for this process
        avg_val_loss = val_loss / len(val_loader)

        # Sync validation loss across all processes
        val_loss_tensor = torch.tensor([avg_val_loss], device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = val_loss_tensor.item() / world_size

        # Log metrics (only on main process)
        if is_main_process(rank):
            logger.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            }, epoch)

            print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Check if this is the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = f"{config.model_save_path}/sentence_transformer"
                utils.create_directory(model_path)

                # Save the unwrapped model (without DDP)
                sentence_transformer.module.save_pretrained(model_path)
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")

        # Synchronize best_val_loss across processes
        best_val_loss_tensor = torch.tensor([best_val_loss], device=device)
        dist.all_reduce(best_val_loss_tensor, op=dist.ReduceOp.MIN)
        best_val_loss = best_val_loss_tensor.item()

        # Make sure all processes have a consistent view of the best model
        dist.barrier()

    # Final cleanup
    if is_main_process(rank):
        logger.close()

    # Wait for all processes to finish
    dist.barrier()

    # Return the unwrapped model
    return sentence_transformer.module


def prepare_reasoning_pairs_dataset_distributed(model_name, queries, max_pairs=1000, rank=0):
    """
    Prepare a dataset of original and condensed reasoning pairs in a distributed setting

    Args:
        model_name: Name of the model to generate reasoning
        queries: List of queries to generate reasoning for
        max_pairs: Maximum number of pairs to generate
        rank: Process rank (only rank 0 will generate the dataset)

    Returns:
        Dataset of reasoning pairs
    """
    # Only the main process (rank 0) generates the dataset
    if rank == 0:
        # Limit number of queries to process
        queries = queries[:max_pairs]

        # Clear CUDA cache before loading model
        torch.cuda.empty_cache()
        gc.collect()

        # Create a simple wrapper class for ReasoningPairsGenerator that
        # handles smaller batches to avoid OOM errors
        class MemoryEfficientReasoningGenerator:
            def __init__(self, model_name):
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                # Use CPU first to avoid OOM during initialization
                self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", max_memory={0: "70GiB", 1: "6GiB", 2: "70GiB", 3: "70GiB"})
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token

                # Set device based on available memory
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {self.device}")

            def create_dataset(self, queries, batch_size=1):
                import torch
                from tqdm import tqdm

                all_pairs = []

                # Process queries in smaller batches
                for i in range(0, len(queries), batch_size):
                    batch_queries = queries[i:i+batch_size]

                    for query in tqdm(batch_queries, desc="Generating reasoning pairs"):
                        # Generate original reasoning with CoT prompt
                        cot_prompt = f"Question: {query}\nLet's think through this step by step to find the answer."

                        inputs = self.tokenizer(cot_prompt, return_tensors="pt").to(self.device)

                        with torch.no_grad():
                            outputs = self.model.generate(
                                inputs.input_ids,
                                max_length=512,
                                temperature=0.7,
                                do_sample=True,
                                top_p=0.9
                            )

                        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        original_reasoning = full_response.replace(cot_prompt, "").strip()

                        # Generate condensed reasoning with a different prompt
                        condense_prompt = f"Question: {query}\nProvide a brief, condensed chain of reasoning to answer this question."

                        inputs = self.tokenizer(condense_prompt, return_tensors="pt").to(self.device)

                        with torch.no_grad():
                            outputs = self.model.generate(
                                inputs.input_ids,
                                max_length=256,  # Shorter for condensed reasoning
                                temperature=0.7,
                                do_sample=True,
                                top_p=0.9
                            )

                        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        condensed_reasoning = full_response.replace(condense_prompt, "").strip()

                        # Create a pair
                        pair = {
                            "original_reasoning": original_reasoning,
                            "condensed_reasoning": condensed_reasoning
                        }

                        all_pairs.append(pair)

                        # Clear cache after each query
                        torch.cuda.empty_cache()

                return all_pairs

        # Generate reasoning pairs with memory-efficient generator
        print(f"Generating {len(queries)} reasoning pairs...")
        generator = MemoryEfficientReasoningGenerator(model_name)
        dataset = generator.create_dataset(queries, batch_size=1)  # Process one query at a time

        return dataset
    else:
        # Other processes will wait for the main process to create the dataset
        return None