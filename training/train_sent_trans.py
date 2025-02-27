import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.sentence_transformer import CustomizedSentenceTransformer
from transformers import AutoModel, AutoTokenizer
from data.reasoning_pairs import ReasoningPairsGenerator
from utils.logging import Logger
import utils.utils as utils

def train_sentence_transformer(
    base_model_name,
    start_layer_idx,
    end_layer_idx,
    dataset,
    config
):
    """
    Train a customized sentence transformer to measure similarity between
    reasoning pairs (original and condensed reasoning)

    Args:
        base_model_name: Name of the base model
        start_layer_idx: Start layer for extraction
        end_layer_idx: End layer for extraction
        dataset: Dataset containing reasoning pairs
        config: Training configuration
    """
    device = config.device

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
    sentence_transformer = sentence_transformer.to(device)

    # Define optimizer
    optimizer = optim.AdamW(
        sentence_transformer.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Define contrastive loss
    contrastive_loss = nn.CosineEmbeddingLoss(margin=0.2)

    # Setup logger
    logger = Logger(
        log_dir=config.log_dir,
        experiment_name=f"sentence_transformer_{start_layer_idx}_to_{end_layer_idx}"
    )
    logger.log_hyperparams(config.__dict__)

    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        # Training phase
        sentence_transformer.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Training"):
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

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        sentence_transformer.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Validation"):
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

        avg_val_loss = val_loss / len(val_loader)

        # Log metrics
        logger.log_metrics({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        }, epoch)

        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = f"{config.model_save_path}/sentence_transformer"
            utils.create_directory(model_path)
            sentence_transformer.save_pretrained(model_path)

            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

    logger.close()
    return sentence_transformer

def prepare_reasoning_pairs_dataset(model_name, queries, max_pairs=1000):
    """
    Prepare a dataset of original and condensed reasoning pairs

    Args:
        model_name: Name of the model to generate reasoning
        queries: List of queries to generate reasoning for
        max_pairs: Maximum number of pairs to generate

    Returns:
        Dataset of reasoning pairs
    """
    # Initialize reasoning pairs generator
    pairs_generator = ReasoningPairsGenerator(model_name)

    # Limit number of queries to process
    queries = queries[:max_pairs]

    # Generate reasoning pairs
    print(f"Generating {len(queries)} reasoning pairs...")
    dataset = pairs_generator.create_dataset(queries)

    return dataset

if __name__ == "__main__":
    # This allows running the script directly for testing
    import argparse
    from config.model_config import ModelConfig
    from config.experiment_config import ExperimentConfig

    parser = argparse.ArgumentParser(description="Train Sentence Transformer")
    parser.add_argument("--config", type=str, default="default", help="Configuration name")
    args = parser.parse_args()

    model_config = ModelConfig(args.config)
    experiment_config = ExperimentConfig(args.config)

    # Load GSM8K dataset for queries
    from data.datasets import load_gsm8k_dataset
    train_dataset, _ = load_gsm8k_dataset(model_config.data_path)

    # Extract queries from the dataset
    queries = [item["query"] for item in train_dataset][:100]  # Limit for testing

    # Prepare reasoning pairs dataset
    pairs_dataset = prepare_reasoning_pairs_dataset(
        model_config.teacher_model_name,
        queries,
        max_pairs=experiment_config.max_reasoning_pairs
    )

    # Train sentence transformer
    sentence_transformer = train_sentence_transformer(
        model_config.teacher_model_name,
        experiment_config.start_layer_idx,
        experiment_config.end_layer_idx,
        pairs_dataset,
        experiment_config
    )