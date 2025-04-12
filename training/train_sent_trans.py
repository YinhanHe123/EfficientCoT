import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.sentence_transformer import CustomizedSentenceTransformer
from transformers import AutoModel, AutoTokenizer
from data.gpt4pair import ReasoningPairsGenerator
from utils.logging import Logger
import utils.utils as utils
import numpy as np
import os # DEBUG

def contrastive_loss(original_embeddings, condensed_embeddings):
    """
    Compute contrastive loss between original and condensed embeddings

    Args:
        original_embeddings: Embeddings for original reasoning
        condensed_embeddings: Embeddings for condensed reasoning

    Returns:
        Contrastive loss value
    """
    # Compute similarity matrix
    norm_original = torch.norm(original_embeddings, dim=1, keepdim=True)
    norm_condensed = torch.norm(condensed_embeddings, dim=1, keepdim=True)
    norm_mat = torch.matmul(norm_original, norm_condensed.T)
    sim_mat = torch.matmul(original_embeddings, condensed_embeddings.T)/ norm_mat
    # Contrastive loss calculation

    loss = -torch.mean(
        torch.log(
            torch.exp(torch.diag(sim_mat)) / torch.sum(torch.exp(sim_mat), dim=1, keepdim=False)
        )
    )
    return loss

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
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to end of sequence token
    # tokenizer.pad_token = tokenizer.eos_token

    # Freeze the base model parameters
    for name, param in base_model.named_parameters():
        param.requires_grad = False


    # Initialize sentence transformer
    sentence_transformer = CustomizedSentenceTransformer(
        base_model_name,
        start_layer_idx,
        end_layer_idx,
        config.embedding_dim
    )
    sentence_transformer = sentence_transformer.to(device)
    # ---------------DEBUG START--------------------
    # sentence_transformer_old = CustomizedSentenceTransformer(
    #     base_model_name,
    #     start_layer_idx,
    #     end_layer_idx,
    #     config.embedding_dim
    # )

    # sentence_transformer_old.load_state_dict(torch.load('/data/nee7ne/effi_cot/saved_models/effi_cot/old_vanilla/sentence_transformer/model.pt', map_location='cpu'))
    # sentence_transformer_old = sentence_transformer_old.to(device)
    # ---------------DEBUG END--------------------
    # Define optimizer


    optimizer = optim.AdamW(
        sentence_transformer.parameters(),
        lr=config.sent_trans_epochs,
        weight_decay=config.sent_trans_weight_decay
    )

    # Define contrastive loss

    # Setup logger
    logger = Logger(
        log_dir=config.log_dir,
        experiment_name=f"sentence_transformer_{start_layer_idx}_to_{end_layer_idx}"
    )
    logger.log_hyperparams(config.__dict__)

    # Split dataset into train and validation
    generator = torch.Generator().manual_seed(config.seed)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=lambda worker_id: np.random.seed(config.seed + worker_id),
        generator=generator
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config.sent_trans_epochs):
        # Training phase
        sentence_transformer.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.train_sen_trans_epochs} - Training"):
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

            loss = contrastive_loss(original_embeddings, condensed_embeddings)
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

                # Compute loss
                loss = contrastive_loss(original_embeddings, condensed_embeddings)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Log metrics
        logger.log_metrics({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        }, epoch)

        print(f"Epoch {epoch+1}/{config.train_sen_trans_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # -------------------DEBUG START--------------------
        # Save best model
        # os.makedirs(f"{config.model_save_path}/sentence_transformer_ckpts", exist_ok=True)
        # # Save model weights
        # torch.save(sentence_transformer.state_dict(), os.path.join(f"{config.model_save_path}/sentence_transformer_ckpts", f"model_epoch_{epoch}.pt"))
        # ----------------DEBUG END--------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = f"{config.model_save_path}/sentence_transformer"
            utils.create_directory(model_path)
            sentence_transformer.save_pretrained(model_path)

            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        # ----------------DEBUG START--------------------
        # compare with old model
        # with torch.no_grad():
        #     gap = compare_model_parameters(sentence_transformer, sentence_transformer_old)
        #     print(f"Max parameter gap: {gap:.6f}")
        # -----------------DEBUG END--------------------
    logger.close()
    return sentence_transformer


# -----------------DEBUG START--------------------
def compare_model_parameters(model_a, model_b):
    """Compares the parameters of two PyTorch models."""

    # Check if the models have the same parameters
    # Load the models

    if set(model_a.keys()) != set(model_b.keys()):
        return False
    gap = 0
    # Iterate through the parameters and compare them
    for param_name in model_a.keys():
        param_a = model_a[param_name]
        param_b = model_b[param_name]

        # Check if the parameters are equal
        print("model_a device", param_a.device)
        print("model_b device", param_b.device)
        param_a = param_a.cpu()
        param_b = param_b.cpu()

        gap += torch.abs(param_a - param_b).max()

    return gap
# -----------------DEBUG END--------------------

def prepare_reasoning_pairs_dataset(queries, reasonings, answers, output_path, max_pairs=1000):
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
    pairs_generator = ReasoningPairsGenerator()
    # Limit number of queries to process
    queries = queries[:max_pairs]
    reasonings = reasonings[:max_pairs] if reasonings else None
    answers = answers[:max_pairs]
    # Generate reasoning pairs
    print(f"Generating {len(queries)} reasoning pairs...")
    dataset = pairs_generator.create_dataset(queries, reasonings, answers, output_path)
    torch.cuda.empty_cache()
    return dataset
