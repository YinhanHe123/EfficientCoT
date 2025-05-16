import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import os
import time

class ContemplationGenerator(nn.Module):
    def __init__(self, student_model_name, teacher_model_name, teacher_hidden_dim, device="cpu", variation="vanilla"):
        super().__init__()
        self.student_model_name = student_model_name
        self.teacher_model_name = teacher_model_name
        self.device = device
        self.variation = variation
        
        # Choose which model to use based on variation
        if variation == "no_small_contemp_gen":
            # Use teacher model with LoRA
            self.model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
            self.model_hidden_dim = self.model.config.hidden_size
            
            # Apply LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,  # rank
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"]  # Target attention layers
            )
            self.model = get_peft_model(self.model, peft_config)
            self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
            
            # No projection needed as we're already using the teacher dimensions
            self.projection_layer = nn.Identity()
        else:
            # Original implementation - use student model
            self.model = AutoModel.from_pretrained(student_model_name)
            self.model_hidden_dim = self.model.config.hidden_size
            self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)
            
            # Create projection layer to match dimensions if needed
            self.projection_layer = nn.Linear(
                self.model_hidden_dim,
                teacher_hidden_dim
            )
        
        self.teacher_hidden_dim = teacher_hidden_dim
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to end of sequence token

    def forward(self, input_ids, attention_mask=None):
        # Generate model hidden states
        stud_start = time.time()
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        stud_end = time.time()
        stud_time = stud_end - stud_start
        
        # Get the last hidden states
        hidden_states = outputs.hidden_states[-1]
        
        # Project to teacher model hidden dimension
        projected_states = self.projection_layer(hidden_states)
        
        return projected_states

    @classmethod
    def from_pretrained(cls, path):
        # Load the model config from the saved path
        config_dict = torch.load(f"{path}/config.pt")

        # Initialize the model with the loaded config
        model = cls(
            config_dict["student_model_name"],
            config_dict["teacher_model_name"],
            config_dict["teacher_hidden_dim"],
            variation=config_dict.get("variation", "vanilla")  # Default to vanilla if not present
        )
        
        # Load the state dict
        model.load_state_dict(torch.load(f"{path}/model.pt", map_location='cpu'))
        
        return model

    def save_pretrained(self, path):
        # Save model config
        config_dict = {
            "student_model_name": self.student_model_name,
            "teacher_model_name": self.teacher_model_name,
            "teacher_hidden_dim": self.teacher_hidden_dim,
            "model_hidden_dim": self.model_hidden_dim,
            "device": self.device,
            "variation": self.variation
        }
        
        # Create path if not exist
        if not os.path.exists(path):
            os.makedirs(path)
            
        torch.save(config_dict, f"{path}/config.pt")

        # Save model weights
        torch.save(self.state_dict(), f"{path}/model.pt")