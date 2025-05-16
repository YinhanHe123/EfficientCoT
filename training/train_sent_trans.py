import random
import torch
import torch.optim as optim
from tqdm import tqdm
from models.sentence_transformer import CustomizedSentenceTransformer
from transformers import AutoModel, AutoTokenizer
from data.gpt4pair import ReasoningPairsGenerator
from utils.logging import Logger
import utils.utils as utils

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
    # Setup logger
    logger = Logger(
        log_dir=config.log_dir,
        experiment_name=config.experiment_name
    )
    logger.logger.info("Training sentence transformer")
    
    device = config.device

    # Initialize the full base model for getting hidden states
    base_model = AutoModel.from_pretrained(base_model_name).to(device)
    base_model.eval()  # Keep it in eval mode as we only use it for features
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to end of sequence token

    # Freeze the base model parameters
    for _, param in base_model.named_parameters():
        param.requires_grad = False

    # Initialize sentence transformer
    sentence_transformer = CustomizedSentenceTransformer(
        base_model_name,
        start_layer_idx,
        end_layer_idx,
        config.embedding_dim
    ).to(device)

    for idx, sample in enumerate(tqdm(dataset)):
        original_inputs = tokenizer(
            sample["reasoning"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_seq_length
        ).to(device)

        condensed_inputs = tokenizer(
            sample["condensed_reasoning"],
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
            dataset.update_item(idx, "gt_reason_hidden", original_outputs.hidden_states[start_layer_idx].cpu())
            dataset.update_item(idx, "condensed_reason_hidden", condensed_outputs.hidden_states[start_layer_idx].cpu())
        del original_inputs, condensed_inputs, original_outputs, condensed_outputs
        torch.cuda.empty_cache()

    # Split dataset into train and validation
    generator = torch.Generator().manual_seed(config.seed)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    for name, param in sentence_transformer.named_parameters():
        if "embedding_projection" not in name:
            param.requires_grad = False
    for (lr, wd, ne) in [(config.st_linear_lr, config.st_linear_wd, config.st_linear_epochs), (config.st_llm_lr, config.st_llm_wd, config.st_llm_epochs)]:
        best_val_loss = float('inf')
        optimizer = optim.AdamW(sentence_transformer.parameters(), lr=lr, weight_decay=wd)
        for epoch in range(ne):
            sentence_transformer.train()
            train_loss, num_batches, batch_orig_embs, batch_condensed_embs = 0, 0, [], []
            for count, idx in enumerate(tqdm(random.sample(range(len(train_dataset)), len(train_dataset)), desc=f"Epoch {epoch+1}/{ne} - Training")):
                original_hidden_states = train_dataset[idx]["gt_reason_hidden"].to(device)
                condensed_hidden_states = train_dataset[idx]["condensed_reason_hidden"].to(device)
                
                batch_orig_embs.append(sentence_transformer(
                    original_hidden_states,
                    attention_mask=torch.ones(original_hidden_states.shape[:2]).to(device)
                ))
                batch_condensed_embs.append(sentence_transformer(
                    condensed_hidden_states,
                    attention_mask=torch.ones(condensed_hidden_states.shape[:2]).to(device)
                ))
                if (count + 1) % config.batch_size == 0:
                    optimizer.zero_grad()
                    batch_loss = contrastive_loss(torch.concat(batch_orig_embs, dim=0), torch.concat(batch_condensed_embs, dim=0))
                    batch_loss.backward()
                    optimizer.step()
                    train_loss += batch_loss.item()
                    num_batches += 1
                    batch_orig_embs, batch_condensed_embs = [], []
            avg_train_loss = train_loss / num_batches
            
            # Validation phase
            sentence_transformer.eval()

            with torch.no_grad():
                val_loss, num_batches, eval_orig_embs, eval_condensed_embs = 0, 0, [], []
                for idx, sample in enumerate(tqdm(val_dataset, desc=f"Epoch {epoch+1}/{ne} - Validation")):
                    original_hidden_states = sample["gt_reason_hidden"].to(device)
                    condensed_hidden_states = sample["condensed_reason_hidden"].to(device)
                    
                    eval_orig_embs.append(sentence_transformer(
                        original_hidden_states,
                        attention_mask=torch.ones(original_hidden_states.shape[:2]).to(device)
                    ))
                    eval_condensed_embs.append(sentence_transformer(
                        condensed_hidden_states,
                        attention_mask=torch.ones(condensed_hidden_states.shape[:2]).to(device)
                    ))
                    if (idx + 1) % config.batch_size == 0:
                        val_loss += contrastive_loss(torch.concat(eval_orig_embs, dim=0), torch.concat(eval_condensed_embs, dim=0)).item()
                        eval_orig_embs, eval_condensed_embs = [], []
                        num_batches += 1
            avg_val_loss = val_loss / num_batches

            # Log metrics
            logger.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            }, epoch)

            print(f"Epoch {epoch+1}/{ne} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = f"{config.model_save_path}/sentence_transformer"
                utils.create_directory(model_path)
                sentence_transformer.save_pretrained(model_path)
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        if ne > 0:
            sentence_transformer = sentence_transformer.from_pretrained(model_path).to(device)
            logger.logger.info(f"Loading best validation loss = {best_val_loss}")
            print(f"Loading best validation loss = {best_val_loss}")
        for param in sentence_transformer.parameters():
            param.requires_grad = True
    logger.close()
    del base_model
    torch.cuda.empty_cache()
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