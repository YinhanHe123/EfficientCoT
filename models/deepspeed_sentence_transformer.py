import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from deepspeed.pipe import PipelineModule, LayerSpec
import copy
import os
import json

class PipelinedSentenceTransformer(nn.Module):
    def __init__(self, base_model_name, start_layer_idx, end_layer_idx, embedding_dim=768):
        super().__init__()

        # Load the base model, tokenizer and config
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to end of sequence token
        self.config = AutoConfig.from_pretrained(base_model_name)

        # Extract layer indices
        self.start_layer_idx = start_layer_idx
        self.end_layer_idx = end_layer_idx
        self.base_model_name = base_model_name

        # Add embedding projection layer for sentence embeddings
        self.embedding_projection = nn.Linear(self.config.hidden_size, embedding_dim)

        # Set up pooling strategy
        self.pooling_strategy = "mean"  # can be "cls", "mean", etc.

    def create_pipeline(self, num_stages=2):
        """
        Create a PipelineModule with the extracted layers
        Args:
            num_stages: Number of pipeline stages
        Returns:
            PipelineModule: DeepSpeed pipeline model
        """
        # Load the base model
        base_model = AutoModel.from_pretrained(self.base_model_name)
        layers = []
        layers.append(base_model.rotary_emb)
        # Extract the required transformer layers
        for i in range(self.start_layer_idx, self.end_layer_idx + 1):
            if i < len(base_model.layers):
                layers.append(base_model.layers[i])
        # Add embedding projection as the final layer
        layers.append(nn.Linear(self.config.hidden_size, self.embedding_projection.out_features))

        # Create PipelineModule
        pipe_model = PipelineModule(
            layers=layers,
            num_stages=num_stages,
            loss_fn=torch.nn.MSELoss()  # Placeholder loss function
        )
        return pipe_model

    def forward(self, hidden_states, attention_mask=None):
        """This forward will be used for non-pipelined inference"""
        device = hidden_states.device
        batch_size, seq_length, _ = hidden_states.shape

        # Load the base model for inference
        base_model = AutoModel.from_pretrained(self.base_model_name).to(device)

        # Extract the layers
        norm = copy.deepcopy(base_model.norm).to(device)

        # We need to apply only the relevant layers starting from the hidden_states
        layers = []
        for i in range(self.start_layer_idx, self.end_layer_idx + 1):
            if i < len(base_model.layers):
                layers.append(copy.deepcopy(base_model.layers[i]).to(device))

        # Forward through the layers
        for layer in layers:
            # Pass through layer
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask
            )

            # LLaMA layers return a tuple; first element is the hidden states
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs

        # Apply final norm layer
        hidden_states = norm(hidden_states)

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

    def encode_text(self, texts, max_length=512):
        """Encode text inputs to get hidden states that can be passed to forward"""
        # This method is not used in pipelined mode
        raise NotImplementedError("encode_text is not implemented for pipelined mode")

    @classmethod
    def from_pretrained(cls, path):
        """Load a pre-trained sentence transformer"""
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

        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        return model

    def save_pretrained(self, path):
        """Save the model to disk"""
        os.makedirs(path, exist_ok=True)

        # Save configuration
        config = {
            "base_model_name": self.base_model_name,
            "start_layer_idx": self.start_layer_idx,
            "end_layer_idx": self.end_layer_idx,
            "embedding_dim": self.embedding_projection.out_features
        }

        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(config, f)

        # Save model weights
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))