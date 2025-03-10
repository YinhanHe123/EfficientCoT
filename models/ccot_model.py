import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm

class CCoTModel(nn.Module):
    """
    Implementation of Compressed Chain of Thought model from the paper
    "Compressed Chain of Thought: Efficient Reasoning through Dense Representations"
    by Jeffrey Cheng and Benjamin Van Durme.
    """
    def __init__(self, base_model_name, compression_ratio=0.1, autoregressive_layer=15, device="cuda"):
        """
        Initialize the CCoT model.

        Args:
            base_model_name: Name of the base LLM model
            compression_ratio: Ratio of compressed contemplation tokens to full reasoning chain
            autoregressive_layer: Layer index to use for autoregressive generation of contemplation tokens
            device: Device to run the model on (cuda or cpu)
        """
        super().__init__()
        self.device = device
        self.compression_ratio = compression_ratio
        self.autoregressive_layer = autoregressive_layer

        # Load the base model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create special tokens for CCoT
        self.reasoning_token = "[REASONING]"

        # Add special token if it doesn't exist
        if self.reasoning_token not in self.tokenizer.get_vocab():
            num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": [self.reasoning_token]})
            # Resize token embeddings if new tokens were added
            if num_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))

        # End predictor - determines when to stop generating contemplation tokens
        self.end_predictor = nn.Linear(self.model.config.hidden_size, 1)

        # Move model to the specified device
        self.model = self.model.to(device)
        self.end_predictor = self.end_predictor.to(device)
        self.rotary_emb = self.model.model.rotary_emb
        
    def forward(self, input_ids, attention_mask=None):
        """
        Generate compressed contemplation tokens using autoregressive generation.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for the input

        Returns:
            Contemplation token hidden states
        """
        batch_size = input_ids.size(0)

        # Initial forward pass to get the hidden states for the query
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get the hidden state of the last token at the specified layer
        # This will be used as the first autoregressive token
        last_token_idx = attention_mask.sum(dim=1) - 1
        hidden_states = outputs.hidden_states[self.autoregressive_layer]

        # Extract the last token's hidden state for each item in the batch
        last_token_hidden = torch.stack([
            hidden_states[i, last_token_idx[i]]
            for i in range(batch_size)
        ])

        # Prepare for autoregressive generation of contemplation tokens
        all_contemplation_tokens = [last_token_hidden.unsqueeze(1)]
        max_contemplation_tokens = 50  # Maximum number of contemplation tokens to generate

        # Autoregressively generate contemplation tokens
        for i in range(max_contemplation_tokens):
            # Current token hidden state (for autoregressively generating the next token)
            current_token = all_contemplation_tokens[-1].squeeze(1)

            # Pass through the model to get next token representation
            # We're using the model differently from standard usage - we're directly
            # feeding hidden states as inputs
            extended_hidden_states = torch.cat(
                [hidden_states] + [token for token in all_contemplation_tokens],
                dim=1
            )

            # Get attention mask for the extended sequence
            extended_attention_mask = torch.ones(
                batch_size,
                extended_hidden_states.size(1),
                device=self.device
            )
            extended_attention_mask = None


            # Forward pass through the model using hidden states
            with torch.no_grad():
                # We'll simulate the forward pass through transformer layers
                # starting from the autoregressive layer
                seq_length = extended_hidden_states.size(1)
                # if position_ids is None:
                position_ids = torch.arange(seq_length, device=self.device).unsqueeze(0).expand(batch_size, -1)

                position_embeddings = self.rotary_emb(hidden_states, position_ids)

                current_layer = self.autoregressive_layer
                current_hidden = extended_hidden_states

                while current_layer < self.model.config.num_hidden_layers:
                    # Simulate transformer layer computation
                    # This is a simplified version and depends on the specific model architecture
                    next_hidden = self.model.model.layers[current_layer](
                        current_hidden,
                        attention_mask=extended_attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings
                    )[0]
                    current_hidden = next_hidden
                    current_layer += 1

                # Get the last token hidden state
                next_token_hidden = current_hidden[:, -1].unsqueeze(1)

            # Add the generated token to our collection
            all_contemplation_tokens.append(next_token_hidden)

            # Check if we should stop generating (using the end predictor)
            end_pred = torch.sigmoid(self.end_predictor(next_token_hidden.squeeze(1)))
            if (end_pred > 0.5).all():
                break

        # Concatenate all generated contemplation tokens
        contemplation_states = torch.cat(all_contemplation_tokens, dim=1)

        return contemplation_states

    @classmethod
    def from_pretrained(cls, path):
        """
        Load a pretrained CCoT model

        Args:
            path: Path to the saved model

        Returns:
            Loaded CCoT model
        """
        # Load config
        config_path = os.path.join(path, "config.pt")
        config = torch.load(config_path)

        # Initialize model with loaded config
        model = cls(
            config["base_model_name"],
            compression_ratio=config["compression_ratio"],
            autoregressive_layer=config["autoregressive_layer"],
            device=config["device"]
        )

        # Load state dict
        model_path = os.path.join(path, "model.pt")
        model.load_state_dict(torch.load(model_path))

        return model

    def save_pretrained(self, path):
        """
        Save the CCoT model

        Args:
            path: Path to save the model
        """
        os.makedirs(path, exist_ok=True)

        # Save config
        config = {
            "base_model_name": self.model.config._name_or_path,
            "compression_ratio": self.compression_ratio,
            "autoregressive_layer": self.autoregressive_layer,
            "device": self.device
        }
        torch.save(config, os.path.join(path, "config.pt"))

        # Save state dict
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

def train_ccot_model(
    base_model_name,
    train_dataset,
    eval_dataset,
    output_path,
    compression_ratio=0.1,
    autoregressive_layer=15,
    learning_rate=1e-4,
    num_epochs=10,
    batch_size=8,
    device="cuda"
):
    """
    Train a CCoT model using LoRA fine-tuning.

    Args:
        base_model_name: Name of the base model
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        output_path: Path to save the model
        compression_ratio: Ratio of compressed contemplation tokens to full reasoning chain
        autoregressive_layer: Layer index to use for autoregressive generation
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to run the model on

    Returns:
        Trained CCoT model
    """
    from transformers import get_linear_schedule_with_warmup
    import torch.nn.functional as F

    # Initialize model
    model = CCoTModel(
        base_model_name,
        compression_ratio=compression_ratio,
        autoregressive_layer=autoregressive_layer,
        device=device
    )
    model = model.to(device)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset.dataset,
        batch_size=batch_size,
        shuffle=True
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset.dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Set up optimizer and scheduler
    # We only optimize the added parameters for efficient training
    optimizer = torch.optim.AdamW([
        {"params": model.end_predictor.parameters()}
    ], lr=learning_rate)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )

    # Loss function for hidden state similarity
    def hidden_state_loss(pred_states, target_states):
        # Normalize the vectors
        # mean the 1th dimension
        pred_norm = torch.mean(pred_states, dim=1)
        target_norm = torch.mean(target_states, dim=1)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)

        # Loss = 1 - similarity (lower is better)
        return 1 - cosine_sim.mean()

    # Training loop
    best_eval_loss = float('inf')

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            # Extract data
            queries = batch["question"]
            reasonings = batch["answer"] # this is reasoning

            # Tokenize input
            input_ids = model.tokenizer(
                queries,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)

            attention_mask = (input_ids != model.tokenizer.pad_token_id).long()

            # Generate contemplation tokens
            contemplation_states = model(input_ids, attention_mask)

            # Get target hidden states from full reasoning
            with torch.no_grad():
                reasoning_inputs = model.tokenizer(
                    reasonings,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(device)

                reasoning_mask = (reasoning_inputs != model.tokenizer.pad_token_id).long()

                reasoning_outputs = model.model(
                    input_ids=reasoning_inputs,
                    attention_mask=reasoning_mask,
                    output_hidden_states=True,
                    return_dict=True
                )

                # Get the hidden states from the teacher model
                target_states = reasoning_outputs.hidden_states[model.autoregressive_layer]

                # Apply the compression ratio - only keep a subset of the states
                seq_len = target_states.shape[1]
                keep_indices = torch.linspace(
                    0, seq_len-1,
                    steps=int(seq_len * model.compression_ratio),
                    dtype=torch.long,
                    device=device
                )
                compressed_target_states = target_states[:, keep_indices]

            # Compute loss - match the compressed contemplation states to the compressed target states
            loss = hidden_state_loss(
                contemplation_states[:, :compressed_target_states.size(1)],
                compressed_target_states
            )

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # Evaluate
        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for batch in eval_loader:
                # Extract data
                queries = batch["question"]
                reasonings = batch["answer"]

                # Tokenize input
                input_ids = model.tokenizer(
                    queries,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(device)

                attention_mask = (input_ids != model.tokenizer.pad_token_id).long()

                # Generate contemplation tokens
                contemplation_states = model(input_ids, attention_mask)

                # Get target hidden states from full reasoning
                reasoning_inputs = model.tokenizer(
                    reasonings,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(device)

                reasoning_mask = (reasoning_inputs != model.tokenizer.pad_token_id).long()

                reasoning_outputs = model.model(
                    input_ids=reasoning_inputs,
                    attention_mask=reasoning_mask,
                    output_hidden_states=True,
                    return_dict=True
                )

                # Get the hidden states from the teacher model
                target_states = reasoning_outputs.hidden_states[model.autoregressive_layer]

                # Apply the compression ratio
                seq_len = target_states.shape[1]
                keep_indices = torch.linspace(
                    0, seq_len-1,
                    steps=int(seq_len * model.compression_ratio),
                    dtype=torch.long,
                    device=device
                )
                compressed_target_states = target_states[:, keep_indices]

                # Compute loss
                loss = hidden_state_loss(
                    contemplation_states[:, :compressed_target_states.size(1)],
                    compressed_target_states
                )

                eval_loss += loss.item()

        # Log training progress
        avg_train_loss = train_loss / len(train_loader)
        avg_eval_loss = eval_loss / len(eval_loader)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Eval Loss: {avg_eval_loss:.4f}")

        # Save best model
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            # model.save_pretrained(output_path)
            best_state_dict = model.state_dict()
    
    model.load_state_dict(best_state_dict)
    model.save_pretrained(output_path)    
    print(f"  Saved new best model with loss: {best_eval_loss:.4f}")

    return model