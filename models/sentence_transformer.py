import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import copy
import os
import json

class CustomizedSentenceTransformer(nn.Module):
    def __init__(self, base_model_name, start_layer_idx, end_layer_idx, embedding_dim=768):
        super().__init__()

        # Load the base model, tokenizer and config
        base_model = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to end of sequence token
        self.config = AutoConfig.from_pretrained(base_model_name)

        # Extract layer indices
        self.start_layer_idx = start_layer_idx
        self.end_layer_idx = end_layer_idx

        # Get input dimension from the base model
        self.input_dim = base_model.config.hidden_size

        # For LLaMA models, we need to extract specific components
        if hasattr(base_model, 'layers'):  # LLaMA models have this structure
            # Extract the RMS normalization layer and rotary embeddings
            self.norm = copy.deepcopy(base_model.norm)
            self.rotary_emb = copy.deepcopy(base_model.rotary_emb)

            # Extract the required transformer layers
            self.layers = nn.ModuleList()
            for i in range(start_layer_idx, end_layer_idx + 1):
                if i < len(base_model.layers):
                    self.layers.append(copy.deepcopy(base_model.layers[i]))
        else:
            raise ValueError(f"Unsupported model architecture: {base_model_name}. Expected LLaMA-style model.")

        # Add embedding projection layer for sentence embeddings
        self.embedding_projection = nn.Linear(self.input_dim, embedding_dim)

        # Set up pooling strategy
        self.pooling_strategy = "mean"  # can be "cls", "mean", etc.

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, device):
        """
        Prepare attention mask similar to LLaMA implementation
        """
        # Create causal mask
        if attention_mask is not None:
            return attention_mask  # Use provided mask

        batch_size, seq_length = input_shape

        # Causal mask is not needed for sentence encoding
        # Just create a simple attention mask (all ones) to allow full attention
        return torch.ones((batch_size, seq_length), device=device)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        """
        Process hidden states through the extracted LLaMA layers

        Args:
            hidden_states: Input hidden states (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask for the sequence (batch_size, seq_len)
            position_ids: Optional position IDs (batch_size, seq_len)

        Returns:
            Tensor: Sentence embeddings
        """
        device = hidden_states.device
        batch_size, seq_length, _ = hidden_states.shape
        # batch_size, seq_length = hidden_states.shape

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

        # Prepare attention mask
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), device
        )
        attention_mask = None

        # Generate position embeddings - similar to LLaMA
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Pass through the extracted layers
        for layer in self.layers:
            # Process through LLaMA layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )

            # LLaMA layers return a tuple; first element is the hidden states
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        # Apply normalization
        hidden_states = self.norm(hidden_states)

        # Pooling strategy
        if self.pooling_strategy == "cls":
            # Use first token
            pooled_output = hidden_states[:, 0]
        else:  # Mean pooling
            # Apply attention mask for proper mean pooling
            if attention_mask is not None:
                if attention_mask.dim() < hidden_states.dim():
                    expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                else:
                    expanded_mask = attention_mask

                # Apply mask and compute mean
                masked_hidden = hidden_states * expanded_mask
                sum_mask = expanded_mask.sum(dim=1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
                pooled_output = masked_hidden.sum(dim=1) / sum_mask
            else:
                # Simple mean pooling
                pooled_output = hidden_states.mean(dim=1)

        # Project to embedding space
        sentence_embedding = self.embedding_projection(pooled_output)

        return sentence_embedding

    def compute_similarity(self, sent1_embedding, sent2_embedding):
        """Compute cosine similarity between two sentence embeddings"""
        return torch.nn.functional.cosine_similarity(
            sent1_embedding, sent2_embedding, dim=-1
        )

    def encode_text(self, texts, max_length=120):
        """
        Encode text inputs to get hidden states that can be passed to forward
        """
        # Tokenize the texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(next(self.parameters()).device)

        # Get the base model to process the text
        base_model = AutoModel.from_pretrained(self.tokenizer.name_or_path)
        base_model = base_model.to(next(self.parameters()).device)
        base_model.eval()

        with torch.no_grad():
            # Get hidden states from base model
            outputs = base_model(
                **inputs,
                output_hidden_states=True
            )

            # Get the hidden states from the start_layer_idx
            hidden_states = outputs.hidden_states[self.start_layer_idx]

        # Process through our extracted layers
        return self.forward(
            hidden_states,
            attention_mask=inputs.attention_mask,
            position_ids=None  # Will be auto-generated
        )

    @classmethod
    def from_pretrained(cls, path):
        """
        Load a pre-trained sentence transformer
        """
        # Load configuration
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Initialize model
        model = cls(
            config["base_model_name"],
            config["start_layer_idx"],
            config["end_layer_idx"],
            config["embedding_dim"]
        )

        # Load model weights
        model_path = os.path.join(path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        params_before = {}
        for name, param in model.named_parameters():
            params_before[name] = param.clone().detach().cpu()

        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        # model.load_state_dict(torch.load('/data/nee7ne/effi_cot/saved_models/effi_cot/old_vanilla/sentence_transformer/model.pt', map_location='cpu'))

        for name, param in model.named_parameters():
            # Check if parameter has changed
            if not torch.allclose(params_before[name], param.cpu()):
                print(f"Parameter {name} has changed")
            else:
                print(f"Parameter {name} is unchanged")

        return model

    def save_pretrained(self, path):
        """
        Save the model to disk
        """
        os.makedirs(path, exist_ok=True)

        # Save configuration
        config = {
            "base_model_name": self.tokenizer.name_or_path,
            "start_layer_idx": self.start_layer_idx,
            "end_layer_idx": self.end_layer_idx,
            "embedding_dim": self.embedding_projection.out_features
        }

        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(config, f)

        # Save model weights
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))