import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
from peft import PeftModelForCausalLM

class CoconutModel(nn.Module):
    """
    Implementation of Chain of Continuous Thought (Coconut) model from the paper
    "Training Large Language Models to Reason in a Continuous Latent Space"
    by Shibo Hao et al.
    """
    def __init__(self, base_model_name, device="cuda"):
        """
        Initialize the Coconut model.

        Args:
            base_model_name: Name of the base LLM model
            device: Device to run the model on (cuda or cpu)
        """
        super().__init__()
        self.device = device
        self.base_model_name = base_model_name

        # Load the base model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create special tokens for Coconut
        self.bot_token = "<bot>"  # Beginning of thought token
        self.eot_token = "<eot>"  # End of thought token

        # Add special tokens if they don't exist
        if self.bot_token not in self.tokenizer.get_vocab():
            special_tokens = {"additional_special_tokens": [self.bot_token, self.eot_token]}
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            # Resize token embeddings if new tokens were added
            if num_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))

        # Move model to the specified device
        self.model = self.model.to(device)

    def forward(self, input_ids, attention_mask=None, continuous_thoughts=None, max_continuous_tokens=5):
        """
        Perform Coconut reasoning with continuous thought tokens.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for the input
            continuous_thoughts: Optional pre-existing continuous thoughts
            max_continuous_tokens: Maximum number of continuous thought tokens to generate

        Returns:
            Model outputs
        """
        batch_size = input_ids.size(0)

        # Initial forward pass to get the hidden states for the query
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Find positions of <bot> tokens
        bot_token_id = self.tokenizer.convert_tokens_to_ids(self.bot_token)
        bot_positions = (input_ids == bot_token_id).nonzero(as_tuple=True)[1]

        # If no <bot> token found or continuous_thoughts is provided, just return outputs
        if len(bot_positions) == 0 or continuous_thoughts is not None:
            return outputs

        # Get last hidden state for the token before <bot>
        bot_pos = bot_positions[0].item()
        last_token_hidden = outputs.hidden_states[-1][:, bot_pos-1, :]
        last_hidden_sequence = outputs.hidden_states[-1]
        # Store hidden states for continuous thoughts
        all_continuous_thoughts = [last_token_hidden.unsqueeze(1)]
        continuous_thoughts = last_token_hidden.unsqueeze(1)

        # Autoregressively generate continuous thought tokens
        for i in range(max_continuous_tokens):
            # Current token hidden state
            # current_token = all_continuous_thoughts[-1].squeeze(1)


            # Pass through the model layers
            # layer_input = current_token.unsqueeze(1)  # [batch_size, 1, hidden_size]

            layer_input = torch.cat((last_hidden_sequence, continuous_thoughts), dim=1)

            # Initialize position embeddings
            seq_length = layer_input.size(1)
            position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1).to(self.device)

            # Get embedding layer
            embedding_layer = self.model.get_input_embeddings()


            # Generate position embeddings - similar to LLaMA
            if isinstance(self.model, PeftModelForCausalLM):
                model_conductor = self.model.base_model.model.model
            else:
                model_conductor = self.model.model
            position_embeddings = model_conductor.rotary_emb(
                layer_input,
                position_ids
            ) if hasattr(model_conductor, 'rotary_emb') else None
            # Forward pass through model layers
            for layer in self.model.model.layers:
                layer_output = layer(
                    layer_input,
                    attention_mask=None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )[0]
                layer_input = layer_output

            # Apply final normalization
            if hasattr(self.model.model, 'norm'):
                next_token_hidden = self.model.model.norm(layer_output)
            else:
                next_token_hidden = layer_output

            # Add to continuous thoughts
            all_continuous_thoughts.append(next_token_hidden[:, -1, :].unsqueeze(1))

        # Concatenate all continuous thoughts
        continuous_thoughts = torch.cat(all_continuous_thoughts, dim=1)

        return continuous_thoughts

    def generate_with_continuous_thoughts(self, input_text, max_continuous_tokens=5, max_new_tokens=30, temperature=0.7):
        """
        Generate an answer using continuous thought reasoning.

        Args:
            input_text: Input text/query
            max_continuous_tokens: Maximum number of continuous thought tokens to generate
            max_new_tokens: Maximum number of new tokens to generate for the answer

        Returns:
            Generated text
        """
        # Add <bot> token to input text to trigger continuous thought mode
        if self.bot_token not in input_text:
            input_text = f"{input_text} {self.bot_token}"

        # Tokenize the input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate continuous thoughts
        continuous_thoughts = self(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_continuous_tokens=max_continuous_tokens
        )

        # Get the embeddings from the model's embedding layer
        input_embeds = self.model.get_input_embeddings()(inputs.input_ids)

        # Add <eot> token to indicate end of continuous thoughts
        eot_token_id = self.tokenizer.convert_tokens_to_ids(self.eot_token)
        eot_input_ids = torch.tensor([[eot_token_id]]).to(self.device)
        eot_embeds = self.model.get_input_embeddings()(eot_input_ids)

        # Concatenate input embeddings, continuous thoughts, and <eot> token
        combined_embeds = torch.cat([
            input_embeds,
            continuous_thoughts,
            eot_embeds.expand(input_embeds.size(0), -1, -1)
        ], dim=1)

        # Create attention mask for combined input
        combined_attention_mask = torch.ones(
            (inputs.input_ids.size(0), combined_embeds.size(1)),
            dtype=torch.long,
            device=self.device
        )

        # Generate text with continuous thoughts
        outputs = self.model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode only the generated text (after <eot>)
        eot_position = combined_embeds.size(1) if outputs.shape[-1] > combined_embeds.size(1) else 0
        generated_text = self.tokenizer.decode(
            outputs[0][eot_position:],
            skip_special_tokens=True
        )

        return generated_text

    @classmethod
    def from_pretrained(cls, path):
        """
        Load a pretrained Coconut model

        Args:
            path: Path to the saved model

        Returns:
            Loaded Coconut model
        """
        # Load config
        config_path = os.path.join(path, "config.pt")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config = torch.load(config_path)

        # Initialize model with loaded config
        model = cls(
            config["base_model_name"],
            device=config["device"]
        )

        # Load state dict
        model_path = os.path.join(path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model.load_state_dict(torch.load(model_path))

        return model

    def save_pretrained(self, path):
        """
        Save the Coconut model

        Args:
            path: Path to save the model
        """
        os.makedirs(path, exist_ok=True)

        # Save config
        config = {
            "base_model_name": self.base_model_name,
            "device": self.device
        }
        torch.save(config, os.path.join(path, "config.pt"))

        # Save state dict
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))