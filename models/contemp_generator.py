import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class ContemplationGenerator(nn.Module):
    def __init__(self, student_model_name, teacher_model_name, teacher_hidden_dim):
        super().__init__()
        self.student_model = AutoModel.from_pretrained(student_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

        # Add a projection layer to match teacher's hidden dimension
        self.student_hidden_dim = self.student_model.config.hidden_size
        self.teacher_hidden_dim = teacher_hidden_dim

        # Create projection layer to match dimensions
        self.projection_layer = nn.Linear(
            self.student_hidden_dim,
            self.teacher_hidden_dim
        )

    def forward(self, input_ids, attention_mask=None):
        # Generate student model hidden states
        outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get the last hidden states
        hidden_states = outputs.last_hidden_state

        # Project to teacher model hidden dimension
        projected_states = self.projection_layer(hidden_states)

        return projected_states

    def generate_contemplation_tokens(self, query, max_length=256):
        # Tokenize the query
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.student_model.device)

        # Generate hidden states
        with torch.no_grad():
            hidden_states = self(
                inputs.input_ids,
                attention_mask=inputs.attention_mask
            )

        return hidden_states

    @classmethod
    def from_pretrained(cls, path):
        # Load the model config from the saved path
        config_dict = torch.load(f"{path}/config.pt")

        # Initialize the model with the loaded config
        model = cls(
            config_dict["student_model_name"],
            config_dict["teacher_model_name"],
            config_dict["teacher_hidden_dim"]
        )

        # Load the state dict
        model.load_state_dict(torch.load(f"{path}/model.pt"))

        return model

    def save_pretrained(self, path):
        # Save model config
        config_dict = {
            "student_model_name": self.student_model.config._name_or_path,
            "teacher_model_name": self.tokenizer.name_or_path,
            "teacher_hidden_dim": self.teacher_hidden_dim,
            "student_hidden_dim": self.student_hidden_dim
        }
        torch.save(config_dict, f"{path}/config.pt")

        # Save model weights
        torch.save(self.state_dict(), f"{path}/model.pt")