import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class SoftCoTModel(nn.Module):
    """
    Implementation of Soft Chain of Thought model from the paper
    "SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs"
    by Yige Xu, Xu Guo, Zhiwei Zeng, Chunyan Miao.
    """
    def __init__(self, llm_model_name, assistant_model_name, device="cuda"):
        """
        Initialize the SoftCoT model.

        Args:
            llm_model_name: Name of the backbone LLM
            assistant_model_name: Name of the assistant model to generate soft thought tokens
            device: Device to run the model on (cuda or cpu)
        """
        super().__init__()
        self.device = device
        self.llm_model_name = llm_model_name
        self.assistant_model_name = assistant_model_name

        # Load the LLM model and tokenizer
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # Load the assistant model and tokenizer
        self.assistant_model = AutoModelForCausalLM.from_pretrained(assistant_model_name)
        self.assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model_name)
        self.assistant_tokenizer.pad_token = self.assistant_tokenizer.eos_token

        # Move models to device
        self.assistant_model = self.assistant_model.to(device)

        # Initialize the projection module to map assistant model's hidden states to LLM's space
        # The dimensions will be set when we first use the model
        self.projection_module = None
        self.assistant_hidden_dim = self.assistant_model.config.hidden_size
        self.llm_hidden_dim = None  # Will be set during first use

    def init_projection_module(self):
        """Initialize the projection module with proper dimensions"""
        if self.llm_hidden_dim is None:
            # Try to infer the LLM's hidden dimension from its config
            llm_model = AutoModelForCausalLM.from_pretrained(self.llm_model_name)
            self.llm_hidden_dim = llm_model.config.hidden_size
            del llm_model  # Free up memory
            torch.cuda.empty_cache()

        self.projection_module = nn.Linear(
            self.assistant_hidden_dim,
            self.llm_hidden_dim
        ).to(self.device)

    def generate_soft_thoughts(self, query, num_tokens=5):
        """
        Generate soft thought tokens based on the query using the assistant model.

        Args:
            query: The input query/question
            num_tokens: Number of soft thought tokens to generate

        Returns:
            Soft thought tokens (hidden states from the assistant model)
        """
        # Prepare input with [UNK] tokens as placeholders for soft thoughts
        instruction = "You are helping a large language model solve reasoning problems. Generate helpful thoughts."
        unk_token = self.assistant_tokenizer.unk_token if self.assistant_tokenizer.unk_token else "[UNK]"

        # Construct the input with instruction, query, and UNK tokens
        input_text = f"{instruction}\nQuestion: {query}\n"
        input_text += f"{unk_token} " * num_tokens

        # Tokenize the input
        inputs = self.assistant_tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Adjust as needed
        ).to(self.device)

        # Generate hidden states with the assistant model
        with torch.no_grad():
            outputs = self.assistant_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )

            # Get the last layer hidden states
            hidden_states = outputs.hidden_states[-1]

            # Extract the hidden states corresponding to the UNK tokens
            # Find the positions of UNK tokens in the input
            unk_token_id = self.assistant_tokenizer.convert_tokens_to_ids(unk_token)
            unk_positions = (inputs.input_ids == unk_token_id).nonzero(as_tuple=True)[1]

            # Extract hidden states at those positions
            soft_thought_tokens = hidden_states[:, unk_positions, :]

        return soft_thought_tokens

    def project_soft_thoughts(self, soft_thought_tokens):
        """
        Project the soft thought tokens from the assistant model's space to the LLM's space.

        Args:
            soft_thought_tokens: Soft thought tokens from the assistant model

        Returns:
            Projected soft thought tokens for the LLM
        """
        if self.projection_module is None:
            self.init_projection_module()

        # Project the tokens
        projected_tokens = self.projection_module(soft_thought_tokens)
        return projected_tokens

    def save_pretrained(self, path):
        """
        Save the SoftCoT model (specifically the projection module) to disk.

        Args:
            path: Path to save the model
        """
        os.makedirs(path, exist_ok=True)

        # Save the projection module
        if self.projection_module is not None:
            torch.save(self.projection_module.state_dict(), os.path.join(path, "projection_module.pt"))

        # Save the config
        config = {
            "llm_model_name": self.llm_model_name,
            "assistant_model_name": self.assistant_model_name,
            "assistant_hidden_dim": self.assistant_hidden_dim,
            "llm_hidden_dim": self.llm_hidden_dim,
        }
        torch.save(config, os.path.join(path, "config.pt"))

    @classmethod
    def from_pretrained(cls, path, device="cuda"):
        """
        Load a pretrained SoftCoT model.

        Args:
            path: Path to the saved model
            device: Device to load the model on

        Returns:
            Loaded SoftCoT model
        """
        # Load config
        config_path = os.path.join(path, "config.pt")
        config = torch.load(config_path, map_location=device)

        # Initialize model with loaded config
        model = cls(
            config["llm_model_name"],
            config["assistant_model_name"],
            device=device
        )
        model.llm_hidden_dim = config["llm_hidden_dim"]

        # Initialize and load the projection module
        model.init_projection_module()
        projection_module_path = os.path.join(path, "projection_module.pt")
        model.projection_module.load_state_dict(torch.load(projection_module_path, map_location=device))

        return model