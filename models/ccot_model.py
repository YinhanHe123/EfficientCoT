import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import os
from tqdm import tqdm
import copy
import pdb
from peft.peft_model import PeftModelForCausalLM

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
        self.base_model_name = base_model_name
        self.compression_ratio = compression_ratio
        self.autoregressive_layer = autoregressive_layer

        # Load the base model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create special tokens for CCoT
        self.reasoning_token = "[REASONING]"
        self.end_token = "[END]"

        # Add special tokens if they don't exist
        if self.reasoning_token not in self.tokenizer.get_vocab():
            special_tokens = {"additional_special_tokens": [self.reasoning_token, self.end_token]}
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            # Resize token embeddings if new tokens were added
            if num_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))

        # End predictor - determines when to stop generating contemplation tokens
        self.end_predictor = nn.Linear(self.model.config.hidden_size, 1)

        # Move model to the specified device
        self.model = self.model.to(device)
        self.end_predictor = self.end_predictor.to(device)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, target_states=None): #target_states is None is set to avoid 'target_states' signiture removal when data is passed to trainer()
        """
        Generate compressed contemplation tokens using autoregressive generation.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for the input
            output_hidden_states: Whether to return all hidden states

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
        for _ in range(max_contemplation_tokens):
            # Current token hidden state (for autoregressively generating the next token)
            current_token = all_contemplation_tokens[-1].squeeze(1)

            # Pass the current token through the transformer layers starting from autoregressive_layer
            layer_input = current_token.unsqueeze(1)  # Shape: [batch_size, 1, hidden_size]
            
            # Forward pass through remaining layers
            layer_output = layer_input
            seq_length = layer_output.size(1)
            position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1).to(hidden_states.device)
            attention_mask = None

            # Generate position embeddings - similar to LLaMA
            if isinstance(self.model, PeftModelForCausalLM):
                model_conductor = self.model.base_model.model.model
            else:
                model_conductor = self.model.model
            position_embeddings = model_conductor.rotary_emb(hidden_states, position_ids)
            for i in range(self.autoregressive_layer, len(model_conductor.layers)):
                layer = model_conductor.layers[i]
                layer_output = layer(layer_output,
                                     attention_mask=attention_mask,
                                     position_ids=position_ids,
                                     position_embeddings=position_embeddings)[0]
                
                
            # position_embeddings = self.model.base_model.model.model.rotary_emb(hidden_states, position_ids)

            # for i in range(self.autoregressive_layer, len(self.model.base_model.model.model.layers)):
            #     layer = self.model.base_model.model.model.layers[i]
            #     layer_output = layer(layer_output,
            #                             attention_mask=None,
            #                             position_ids=position_ids,
            #                             position_embeddings=position_embeddings)[0]
            
            # Get the next token hidden state
            next_token_hidden = layer_output.squeeze(1).unsqueeze(1)

            # Add the generated token to our collection
            all_contemplation_tokens.append(next_token_hidden)

            # Check if we should stop generating (using the end predictor)
            end_pred = torch.sigmoid(self.end_predictor(next_token_hidden.squeeze(1)))
            if (end_pred > 0.5).all():
                break

        # Concatenate all generated contemplation tokens
        contemplation_states = torch.cat(all_contemplation_tokens, dim=1)

        if output_hidden_states:
            return contemplation_states, outputs.hidden_states
        return contemplation_states

    def apply_lora_layer_by_layer(self, layer_idx, rank=16, alpha=32):
        """
        Apply LoRA to a specific layer in the model
        
        Args:
            layer_idx: Index of the layer to apply LoRA to
            rank: Rank for LoRA adaptation
            alpha: Alpha parameter for LoRA
            
        Returns:
            Model with LoRA applied to the specified layer
        """
        # Configure LoRA for the specific layer
        target_modules = [f"model.layers.{layer_idx}.self_attn.q_proj", 
                          f"model.layers.{layer_idx}.self_attn.k_proj",
                          f"model.layers.{layer_idx}.self_attn.v_proj",
                          f"model.layers.{layer_idx}.self_attn.o_proj"]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
        )
        
        return get_peft_model(self.model, peft_config)

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
            "base_model_name": self.base_model_name,
            "compression_ratio": self.compression_ratio,
            "autoregressive_layer": self.autoregressive_layer,
            "device": self.device
        }
        torch.save(config, os.path.join(path, "config.pt"))

        # Save state dict
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

# Decode model that uses the generated contemplation tokens
class CCOTDecodeModel(nn.Module):
    """Model to generate answers based on the query and contemplation tokens"""
    def __init__(self, base_model_name, device="cuda"):
        super().__init__()
        self.device = device
        self.base_model_name = base_model_name
        
        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.model = self.model.to(device)
        
    def forward(self, input_ids, attention_mask=None, contemp_states=None, labels=None):
        """
        Generate an answer based on input and contemplation tokens
        
        Args:
            input_ids: Input token IDs for the query
            attention_mask: Attention mask for the input
            contemp_states: Contemplation token states from CCoT model
            labels: Optional labels for computing loss
            
        Returns:
            Model outputs
        """
        # Get the embeddings from the model's embedding layer
        inputs_embeds = self.model.get_input_embeddings()(input_ids).squeeze()
        
        # If contemplation tokens are provided, concatenate them with the input embeddings
        if contemp_states is not None:
            # Limit contemplation states to a reasonable number
            # pdb.set_trace()
            contemp_states = contemp_states.squeeze()
            max_contemp_tokens = min(contemp_states.size(1), 50)
            contemp_to_use = contemp_states[:, :max_contemp_tokens, :]
            print('input_embes', inputs_embeds.size())
            print('contemp_states', contemp_to_use.size())
            
            # Concatenate input embeddings and contemplation states
            combined_embeds = torch.cat([inputs_embeds, contemp_to_use], dim=1)
            
            # Create a new attention mask for the combined sequence
            combined_attention_mask = torch.ones(
                (input_ids.size(0), combined_embeds.size(1)),
                dtype=torch.long,
                device=self.device
            )
            
            # Forward pass with combined embeddings
            outputs = self.model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                # labels=labels,
                return_dict=True
            )
        else:
            # Forward pass with just input embeddings
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # labels=labels,
                return_dict=True
            )
            
        return outputs
    
    def apply_lora(self, rank=16, alpha=32):
        """
        Apply LoRA to the model
        
        Args:
            rank: Rank for LoRA adaptation
            alpha: Alpha parameter for LoRA
            
        Returns:
            Model with LoRA applied
        """
        # Configure LoRA
        # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        target_modules = ["model.layers.15.self_attn.q_proj", 
                        "model.layers.15.self_attn.k_proj",
                        "model.layers.15.self_attn.v_proj",
                        "model.layers.15.self_attn.o_proj"]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            target_modules=target_modules
        )
        
        self.model.enable_input_require_grads()
        self.model = get_peft_model(self.model, peft_config)
        # self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        return self

    @classmethod
    def from_pretrained(cls, path):
        """Load a pretrained Decode model"""
        # Load config
        config_path = os.path.join(path, "config.pt")
        config = torch.load(config_path)

        # Initialize model with loaded config
        model = cls(
            config["base_model_name"],
            device=config["device"]
        )

        # Load state dict
        model_path = os.path.join(path, "model.pt")
        model.load_state_dict(torch.load(model_path))

        return model

    def save_pretrained(self, path):
        """Save the Decode model"""
        os.makedirs(path, exist_ok=True)

        # Save config
        config = {
            "base_model_name": self.base_model_name,
            "device": self.device
        }
        torch.save(config, os.path.join(path, "config.pt"))

        # Save state dict
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        

