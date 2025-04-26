import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import gc

class ImplicitCoTModelWithRNN(nn.Module):
    """
    Implementation of Implicit Chain of Thought with RNN component
    Based on the second project's approach
    """
    def __init__(self, teacher_model_name, student_model_name, hidden_size, device):
        super().__init__()
        self.device = device

        # Save model names
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name

        # Load student model (which will be fine-tuned)
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        self.student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
        self.student_model = self.student_model.to(device)

        # Define hidden size
        self.hidden_size = self.student_model.config.hidden_size

        # RNN component for implicit reasoning
        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=False,
            dropout=0,
            bidirectional=False
        )

        # Projection layers for attention mechanism
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size*2, self.hidden_size)

        # MLP layers for each transformer layer
        num_layers = len(self.student_model.config.hidden_sizes) if hasattr(self.student_model.config, 'hidden_sizes') else 12

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*self.hidden_size, 4*self.hidden_size),
                nn.ReLU(),
                nn.Linear(4*self.hidden_size, self.hidden_size)
            ) for _ in range(num_layers)
        ])

        # Mixture component embeddings
        self.mixture_size = 1  # Default, can be configured
        self.mixture_components = nn.Embedding(self.mixture_size, self.hidden_size)

        # Temperature for softmax
        self.softmax_temperature = 0.05

    def forward(self, input_ids, attention_mask=None, teacher_states=None):
        """
        Forward pass through the model

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            teacher_states: Teacher hidden states (for training)

        Returns:
            Model outputs
        """
        # Get student model embeddings
        outputs = self.student_model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states[-1]  # Last layer's hidden states

        # Create sequence for RNN processing
        batch_size, seq_len, _ = hidden_states.shape

        # Initialize RNN state
        rnn_state = None
        past_keys = None
        context = None

        # Process sequence through RNN
        for i in range(seq_len):
            z = hidden_states[:, i, :]  # Current token's hidden state

            # Get mixture embedding
            if self.mixture_components.weight.shape[0] == 1:
                mixture_embedding = self.mixture_components.weight.expand(batch_size, -1)
            else:
                # Optional: implement mixture selection based on token probabilities
                mixture_embedding = self.mixture_components.weight[0].expand(batch_size, -1)

            # Process through MLP
            f_h_c = self.mlps[0](torch.cat((z, mixture_embedding), dim=-1))

            # Process through RNN
            if context is None:
                context = f_h_c.new_zeros(f_h_c.shape)

            # Add context to input
            rnn_input = (f_h_c + context).unsqueeze(0)

            # Run through RNN
            output, rnn_state = self.rnn(rnn_input, rnn_state)
            output = output.squeeze(0)

            # Create key for attention
            current_key = self.key_proj(output)

            # Attention mechanism for context
            if past_keys is not None:
                current_query = self.query_proj(output)
                attn_weights = torch.bmm(past_keys, current_query.unsqueeze(-1))
                attn_probs = attn_weights.softmax(dim=1)
                attn_probs = attn_probs.squeeze(-1).unsqueeze(1)
                context = torch.bmm(attn_probs, past_keys).squeeze(1)
                past_keys = torch.cat((past_keys, current_key.unsqueeze(1)), dim=1)
            else:
                past_keys = current_key.unsqueeze(1)

            # Output projection
            if i < seq_len - 1:  # Don't need to do this for the last token
                output = self.out_proj(torch.cat((output, context), dim=-1))
                hidden_states[:, i+1, :] = output

        # Final forward pass through the student model with modified hidden states
        logits = self.student_model.lm_head(hidden_states)

        return {"logits": logits, "hidden_states": hidden_states}

    def save_pretrained(self, output_path):
        """Save the model to the specified path"""
        os.makedirs(output_path, exist_ok=True)

        # Save student model
        self.student_model.save_pretrained(f"{output_path}/student_model")
        self.student_tokenizer.save_pretrained(f"{output_path}/student_tokenizer")

        # Save custom components
        torch.save({
            "rnn": self.rnn.state_dict(),
            "key_proj": self.key_proj.state_dict(),
            "query_proj": self.query_proj.state_dict(),
            "out_proj": self.out_proj.state_dict(),
            "mlps": [mlp.state_dict() for mlp in self.mlps],
            "mixture_components": self.mixture_components.state_dict(),
            "softmax_temperature": self.softmax_temperature,
            "teacher_model_name": self.teacher_model_name,
            "student_model_name": self.student_model_name,
            "hidden_size": self.hidden_size,
            "mixture_size": self.mixture_size
        }, f"{output_path}/icot_kd_components.pt")

    @classmethod
    def from_pretrained(cls, path, device="cuda"):
        """Load a pretrained model from the specified path"""
        # Load saved configuration
        config = torch.load(f"{path}/icot_kd_components.pt")

        # Create model instance
        model = cls(
            teacher_model_name=config["teacher_model_name"],
            student_model_name=f"{path}/student_model",
            hidden_size=config["hidden_size"],
            device=device
        )

        # Load custom components
        model.rnn.load_state_dict(config["rnn"])
        model.key_proj.load_state_dict(config["key_proj"])
        model.query_proj.load_state_dict(config["query_proj"])
        model.out_proj.load_state_dict(config["out_proj"])

        for i, mlp_state in enumerate(config["mlps"]):
            if i < len(model.mlps):
                model.mlps[i].load_state_dict(mlp_state)

        model.mixture_components.load_state_dict(config["mixture_components"])
        model.softmax_temperature = config["softmax_temperature"]
        model.mixture_size = config["mixture_size"]

        return model


def train_icot_kd_model(
    teacher_model_name,
    student_model_name,
    train_dataset,
    eval_dataset,
    output_path,
    learning_rate=5e-5,
    num_epochs=3,
    batch_size=4,
    device="cuda"
):
    """
    Train an implicit CoT model with RNN component through knowledge distillation

    Args:
        teacher_model_name: Name of the teacher model
        student_model_name: Name of the student model
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        output_path: Path to save the model
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to run the model on

    Returns:
        Trained model
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load teacher model
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Set to evaluation mode

    # Determine hidden size from the teacher model
    hidden_size = teacher_model.config.hidden_size

    # Initialize the implicit CoT model
    icot_kd = ImplicitCoTModelWithRNN(
        teacher_model_name=teacher_model_name,
        student_model_name=student_model_name,
        hidden_size=hidden_size,
        device=device
    )
    icot_kd = icot_kd.to(device)

    # Freeze teacher model parameters
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Define optimizer
    optimizer = optim.AdamW(icot_kd.parameters(), lr=learning_rate)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=teacher_tokenizer.pad_token_id)

    # Helper functions for prompt formatting
    def format_teacher_prompt(query):
        """Format prompt for teacher to generate CoT reasoning"""
        if "mistral" in teacher_model_name.lower():
            return f"<s>[INST] Question: {query}\n Please solve this step-by-step. [/INST]"
        else:
            return f"<<SYS>>You are an expert in math word problems<</SYS>>\nQuestion: {query}\nPlease solve this step-by-step."

    def format_student_prompt(query):
        """Format prompt for student to generate direct answer"""
        if "mistral" in student_model_name.lower():
            return f"<s>[INST] Question: {query}\n Generate the answer directly. Answer: [/INST]"
        else:
            return f"Question: {query}\n Generate the answer directly. Answer:"

    # Generate teacher hidden states
    def generate_teacher_states(query, answer):
        """Generate teacher hidden states for a given query"""
        # Format prompt with CoT reasoning
        prompt = format_teacher_prompt(query)

        # Tokenize
        inputs = teacher_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        # Generate reasoning with teacher
        with torch.no_grad():
            outputs = teacher_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # Get hidden states from the last layer
            hidden_states = outputs.hidden_states[-1]

            # Get the last token's hidden state
            last_token_idx = inputs.attention_mask.sum(dim=1) - 1
            last_token_hidden = torch.stack([
                hidden_states[i, last_token_idx[i]]
                for i in range(inputs.input_ids.size(0))
            ])

            # Extract representative hidden states
            # For simplicity, we'll use the last token's hidden state
            teacher_state = last_token_hidden

        return teacher_state

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        icot_kd.train()
        total_train_loss = 0

        # Process data in batches
        for i in range(0, len(train_dataset), batch_size):
            batch_data = train_dataset[i:i+batch_size]

            # Extract questions and answers
            questions = [item["query"] for item in batch_data]
            answers = [item["answer"] for item in batch_data]

            # Initialize batch loss
            batch_loss = 0

            # Process each question-answer pair in the batch
            for j, (question, answer) in enumerate(zip(questions, answers)):
                # Generate teacher states
                teacher_state = generate_teacher_states(question, answer)

                # Format student prompt
                prompt = format_student_prompt(question) + " "

                # Tokenize input
                inputs = icot_kd.student_tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)

                # Tokenize target (prompt + answer)
                target_text = prompt  + answer
                target = icot_kd.student_tokenizer(
                    target_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)

                input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
                # Forward pass through implicit CoT model
                while input_ids.shape[-1] < target.input_ids.shape[-1] - 1:
                    outputs = icot_kd(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        teacher_states=teacher_state
                    )
                    input_ids = torch.cat([input_ids, outputs['logits'][0, -1].argmax().reshape(1,1)], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.tensor([[1]]).to(device)], dim=1)
                outputs = icot_kd(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    teacher_states=teacher_state
                )
                # Compute loss
                logits = outputs["logits"]

                # Shift logits and labels for causal language modeling
                shift_logits = logits.contiguous()
                shift_labels = target.input_ids[:, 1:].contiguous()

                # Compute loss
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # Accumulate loss
                batch_loss += loss

            # Average loss over batch
            batch_loss = batch_loss / len(batch_data)

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Accumulate total loss
            total_train_loss += batch_loss.item()

        avg_train_loss = total_train_loss / (len(train_dataset) / batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        icot_kd.eval()
        total_val_loss = 0

        with torch.no_grad():
            for i in range(0, len(eval_dataset), batch_size):
                batch_data = eval_dataset[i:i+batch_size]

                # Extract questions and answers
                questions = [item["query"] for item in batch_data]
                answers = [item["answer"] for item in batch_data]

                # Initialize batch loss
                batch_loss = 0

                # Process each question-answer pair in the batch
                for j, (question, answer) in enumerate(zip(questions, answers)):
                    # Generate teacher states
                    teacher_state = generate_teacher_states(question, answer)

                    # Format student prompt
                    prompt = format_student_prompt(question)

                    # Tokenize input
                    inputs = icot_kd.student_tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(device)

                    # Tokenize target (prompt + answer)
                    target_text = prompt + " " + answer
                    target = icot_kd.student_tokenizer(
                        target_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(device)
                    
                    input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
                    # Forward pass through implicit CoT model
                    while input_ids.shape[-1] < target.input_ids.shape[-1] - 1:
                        outputs = icot_kd(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            teacher_states=teacher_state
                        )
                        input_ids = torch.cat([input_ids, outputs['logits'][0, -1].argmax().reshape(1,1)], dim=1)
                        attention_mask = torch.cat([attention_mask, torch.tensor([[1]]).to(device)], dim=1)
                    outputs = icot_kd(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        teacher_states=teacher_state
                    )

                    # Compute loss
                    logits = outputs["logits"]

                    # Shift logits and labels for causal language modeling
                    shift_logits = logits.contiguous()
                    shift_labels = target.input_ids[:, 1:].contiguous()

                    # Compute loss
                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    # Accumulate loss
                    batch_loss += loss

                # Average loss over batch
                batch_loss = batch_loss / len(batch_data)

                # Accumulate total loss
                total_val_loss += batch_loss.item()

        avg_val_loss = total_val_loss / (len(eval_dataset) / batch_size)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            icot_kd.save_pretrained(output_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

    # Clean up to save memory
    del teacher_model
    gc.collect()
    torch.cuda.empty_cache()

    return icot_kd