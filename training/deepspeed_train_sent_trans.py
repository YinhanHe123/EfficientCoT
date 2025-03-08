import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.deepspeed_sentence_transformer import PipelinedSentenceTransformer
from transformers import AutoModel, AutoTokenizer
from data.reasoning_pairs import ReasoningPairsGenerator
from utils.logging import Logger
import utils.utils as utils
import deepspeed

def train_sentence_transformer_with_deepspeed(
    base_model_name,
    start_layer_idx,
    end_layer_idx,
    dataset,
    config,
    local_rank,
    num_stages=2
):
    """
    Train a customized sentence transformer with DeepSpeed pipeline parallelism

    Args:
        base_model_name: Name of the base model
        start_layer_idx: Start layer for extraction
        end_layer_idx: End layer for extraction
        dataset: Dataset containing reasoning pairs
        config: Training configuration
        local_rank: Local rank for distributed training
        num_stages: Number of pipeline stages
    """
    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed()

    # Create the model
    model = PipelinedSentenceTransformer(
        base_model_name,
        start_layer_idx,
        end_layer_idx,
        config.embedding_dim
    )

    # Create pipelined model
    pipeline_model = model.create_pipeline(num_stages=num_stages)

    # Define DeepSpeed configuration
    ds_config = {
        "train_batch_size": config.batch_size,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": config.batch_size // 2,
        "steps_per_print": 10,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": config.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": config.weight_decay
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

    # Setup logger
    logger = Logger(
        log_dir=config.log_dir,
        experiment_name=f"ds_sentence_transformer_{start_layer_idx}_to_{end_layer_idx}"
    )
    logger.log_hyperparams(config.__dict__)

    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Load base model for extracting hidden states (kept on CPU or local device)
    if model_engine.local_rank == 0:
        base_model = AutoModel.from_pretrained(base_model_name)
        base_model.to(f"cuda:{model_engine.local_rank}")
        base_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token

    # Create data loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=model_engine.global_rank
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,  # This is the micro batch size
        sampler=train_sampler,
        num_workers=2
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config.train_sen_trans_epochs):
        # Set the epoch for the sampler
        train_sampler.set_epoch(epoch)

        # Training phase
        model_engine.train()
        train_loss = 0
        train_samples = 0

        # Process batches of original and condensed reasoning pairs
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.train_sen_trans_epochs} - Training")):
            # Only process batches on rank 0
            if model_engine.local_rank == 0:
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
                ).to(f"cuda:{model_engine.local_rank}")

                condensed_inputs = tokenizer(
                    condensed_reasoning,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_seq_length
                ).to(f"cuda:{model_engine.local_rank}")

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

                    # Transfer to all ranks
                    torch.distributed.broadcast(original_hidden_states, 0)
                    torch.distributed.broadcast(condensed_hidden_states, 0)
            else:
                # Other ranks create placeholder tensors to receive the data
                original_hidden_states = torch.zeros((2, config.max_seq_length, model.student_hidden_dim),
                                                    device=f"cuda:{model_engine.local_rank}")
                condensed_hidden_states = torch.zeros((2, config.max_seq_length, model.student_hidden_dim),
                                                     device=f"cuda:{model_engine.local_rank}")

                # Receive data from rank 0
                torch.distributed.broadcast(original_hidden_states, 0)
                torch.distributed.broadcast(condensed_hidden_states, 0)

            # Forward pass through the pipeline
            loss = model_engine(original_hidden_states, condensed_hidden_states)

            # Backward and optimize
            model_engine.backward(loss)
            model_engine.step()

            # Gather loss from all ranks
            if loss is not None:
                train_loss += loss.item() * original_hidden_states.size(0)
                train_samples += original_hidden_states.size(0)

        # Calculate average loss across all ranks
        if train_samples > 0:
            avg_train_loss = train_loss / train_samples
        else:
            avg_train_loss = 0

        # Log metrics
        if model_engine.global_rank == 0:
            logger.log_metrics({
                "train_loss": avg_train_loss
            }, epoch)

            print(f"Epoch {epoch+1}/{config.train_sen_trans_epochs} - Train Loss: {avg_train_loss:.4f}")

            # Save model checkpoint
            checkpoint_path = os.path.join(config.checkpoint_path, f"ds_sent_trans_epoch{epoch+1}")
            client_state = {"checkpoint_step": epoch}
            model_engine.save_checkpoint(checkpoint_path, client_state=client_state)

    # Save the final sentence transformer model (only on rank 0)
    if model_engine.global_rank == 0:
        # Extract and save the non-pipelined model for easier inference
        sentence_transformer = PipelinedSentenceTransformer(
            base_model_name,
            start_layer_idx,
            end_layer_idx,
            config.embedding_dim
        )

        # Copy weights from the pipeline model
        # This requires custom handling based on how DeepSpeed partitions the model

        model_path = f"{config.model_save_path}/sentence_transformer"
        utils.create_directory(model_path)
        sentence_transformer.save_pretrained(model_path)

        print(f"Saved model to {model_path}")

    logger.close()
    return None  # The model is saved to disk