import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from deepspeed.pipe import PipelineModule, LayerSpec
import os

class PipelinedContemplationGenerator(nn.Module):
    def __init__(self, student_model_name, teacher_model_name, teacher_hidden_dim, device):
        super().__init__()
        self.device = device
        self.student_model_name = student_model_name
        self.teacher_model_name = teacher_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to end of sequence token

        # Get student model config to determine hidden size
        self.student_model = AutoModel.from_pretrained(student_model_name)
        self.student_hidden_dim = self.student_model.config.hidden_size
        self.teacher_hidden_dim = teacher_hidden_dim

        # Create projection layer to match dimensions
        self.projection_layer = nn.Linear(
            self.student_hidden_dim,
            self.teacher_hidden_dim
        )

    def create_pipeline(self, num_stages=2):
        """
        Create a DeepSpeed PipelineModule for the contemplation generator

        Args:
            num_stages: Number of pipeline stages

        Returns:
            PipelineModule: The pipelined model
        """
        # Define layers for pipelining
        layers = [
            # First stage: Student model
            LayerSpec(AutoModel.from_pretrained, self.student_model_name),

            # Second stage: Projection layer
            LayerSpec(nn.Linear, self.student_hidden_dim, self.teacher_hidden_dim)
        ]

        # Create the pipeline module
        pipe_model = PipelineModule(
            layers=layers,
            num_stages=num_stages,
            loss_fn=torch.nn.MSELoss(),  # Placeholder loss function
            partition_method='uniform'  # Split evenly across stages
        )

        return pipe_model

    def forward(self, input_ids, attention_mask=None):
        """
        Regular forward method for non-pipelined inference
        """
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

    @classmethod
    def from_pretrained(cls, path):
        """Load a pre-trained model"""
        # Load the model config from the saved path
        config_dict = torch.load(f"{path}/config.pt", map_location='cpu')

        # Initialize the model with the loaded config
        model = cls(
            config_dict["student_model_name"],
            config_dict["teacher_model_name"],
            config_dict["teacher_hidden_dim"],
            config_dict.get("device", "cuda")
        )

        # Load the state dict
        model.load_state_dict(torch.load(f"{path}/model.pt", map_location='cpu'))

        return model

    def save_pretrained(self, path):
        """Save the model to disk"""
        # Save model config
        config_dict = {
            "student_model_name": self.student_model_name,
            "teacher_model_name": self.teacher_model_name,
            "teacher_hidden_dim": self.teacher_hidden_dim,
            "student_hidden_dim": self.student_hidden_dim,
            "device": self.device
        }
        # create path if not exist
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(config_dict, f"{path}/config.pt")

        # Save model weights
        torch.save(self.state_dict(), f"{path}/model.pt")