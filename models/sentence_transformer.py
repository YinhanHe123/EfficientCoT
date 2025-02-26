import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class CustomizedSentenceTransformer(nn.Module):
    def __init__(self, base_model_name, start_layer_idx, end_layer_idx, embedding_dim=768):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Extract specific layers from the model
        self.start_layer_idx = start_layer_idx
        self.end_layer_idx = end_layer_idx

        # Add embedding projection layer
        self.embedding_projection = nn.Linear(
            self.base_model.config.hidden_size,
            embedding_dim
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Run through base model to get all hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )

        # Extract hidden states from specified layers
        hidden_states = outputs.hidden_states[self.start_layer_idx:self.end_layer_idx+1]

        # Combine hidden states (e.g., by averaging across layers)
        combined_states = torch.stack(hidden_states).mean(dim=0)

        # Get CLS token representation or mean pooling
        if attention_mask is not None:
            # Mean pooling
            token_embeddings = combined_states
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            # CLS token
            pooled_output = combined_states[:, 0]

        # Project to embedding space
        sentence_embedding = self.embedding_projection(pooled_output)

        return sentence_embedding

    def compute_similarity(self, sent1_embedding, sent2_embedding):
        """Compute cosine similarity between two sentence embeddings"""
        return torch.nn.functional.cosine_similarity(
            sent1_embedding, sent2_embedding, dim=-1
        )

    def embed_hidden_states(self, hidden_states, attention_mask=None):
        """
        Process hidden states directly to generate embeddings

        Args:
            hidden_states: Hidden states from a model
            attention_mask: Attention mask for the sequence

        Returns:
            Tensor: Embeddings
        """
        # Process the hidden states as if they come from the matching layer
        # We assume hidden_states are already appropriate for our model

        # Apply mean pooling
        if attention_mask is not None:
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            # Use CLS token or average all tokens
            pooled_output = hidden_states.mean(dim=1)

        # Project to embedding space
        sentence_embedding = self.embedding_projection(pooled_output)

        return sentence_embedding

    @classmethod
    def from_pretrained(cls, path):
        """
        Load a pre-trained sentence transformer

        Args:
            path: Path to the saved model directory

        Returns:
            CustomizedSentenceTransformer: Loaded model
        """
        # Load configuration
        config_path = f"{path}/config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        config = utils.load_json(config_path)

        # Initialize model
        model = cls(
            config["base_model_name"],
            config["start_layer_idx"],
            config["end_layer_idx"],
            config["embedding_dim"]
        )

        # Load model weights
        model_path = f"{path}/model.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model.load_state_dict(torch.load(model_path))

        return model