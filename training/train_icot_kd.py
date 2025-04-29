from logging import Logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import gc

class ModifiedDecoderLayer(nn.Module):
    def __init__(self, original_layer, layer_idx):
        self.layer_idx = layer_idx
        self.orig_layer = original_layer
        self.z = None
        self.f_h_c = None
    
    def reset(self):
        self.z = None
        self.f_h_c = None
    
    def forward(self, hidden_states, attention_mask = None, position_ids = None, past_key_value = None, output_attentions = False, use_cache = False, cache_position = None, position_embeddings = None, **kwargs):
        if "mode" in kwargs and kwargs["mode"] == 'forward_emulator':
            z = hidden_states.gather(1, kwargs["positions_to_take"].view(-1, 1, 1).expand(-1, -1, hidden_states.shape[-1])).squeeze(1) # bsz, hidden_size
            self.z = z
            if kwargs['weight'].shape[0] == 1:
                mixture_embedding = kwargs['weight'].expand(hidden_states.shape[0], -1)
            else:
                log_probs = z @ kwargs['weight'].T # bsz, vocab
                log_probs = log_probs / kwargs['softmax_temperature']
                probs = log_probs.softmax(dim=-1) # bsz, vocab
                mixture_embedding = probs @ kwargs['weight'] # bsz, H
            f_h_c = kwargs['mlp'][self.layer_idx](torch.cat((z, mixture_embedding), dim=-1)) # bsz, hidden_size
            self.f_h_c = f_h_c
            
            if kwargs["rnn"] is not None:
                if kwargs["key_proj"] is not None:
                    if len(kwargs["context"]) == 0:
                        kwargs["context"][-1].append(torch.zeros_like(f_h_c.shape))
                    output, rnn_state = kwargs["rnn"]((f_h_c + kwargs["context"][-1]).unsqueeze(0), kwargs["rnn_state"][-1])
                    kwargs["rnn_state"].append(rnn_state)
                    output = output.squeeze(0)
                    current_key = kwargs["key_proj"](output)
                    if len(kwargs["past_keys"]) > 0:
                        current_query = kwargs["query_proj"](output) # bsz, hidden_size
                        attn_weights = torch.bmm(kwargs["past_keys"][-1], current_query.unsqueeze(-1)) # bsz, len, 1
                        attn_probs = attn_weights.softmax(dim=1)
                        attn_probs = attn_probs.squeeze(-1).unsqueeze(1)
                        kwargs["context"].append(torch.bmm(attn_probs, kwargs["past_keys"][-1]).squeeze(1))
                        kwargs["past_keys"].append(torch.cat((kwargs["past_keys"][-1], current_key.unsqueeze(1)), dim=1))
                    else:
                        kwargs["past_keys"].append(current_key.unsqueeze(1))
                    output = kwargs["out_proj"](torch.cat((output, kwargs["context"][-1]), dim=-1))
                    f_h_c = output
                else:
                    rnn_output, rnn_state = kwargs["rnn"](f_h_c.unsqueeze(0), rnn_state)
                    f_h_c = rnn_output.squeeze(0)
            if kwargs["requires_backward"]:
                hidden_states = hidden_states.clone()
            if kwargs["positions_to_take"].eq(kwargs["positions_to_take"][0]).all():
                hidden_states[:, kwargs["positions_to_take"][0]] = f_h_c
            else:
                for batch_id in range(hidden_states.shape[0]):
                    hidden_states[batch_id, kwargs["positions_to_take"][batch_id]] = f_h_c[batch_id]
        elif "mode" in kwargs and kwargs["mode"] == 'forward_student':
            if kwargs["requires_backward"]:
                hidden_states = hidden_states.clone()
            if kwargs["positions_to_substitute"].eq(kwargs["positions_to_substitute"][0]).all():
                hidden_states[:, kwargs["positions_to_substitute"][0]] = kwargs["states_to_substitute"][self.layer_idx]
            else:
                for batch_id in range(hidden_states.shape[0]):
                    hidden_states[batch_id, kwargs["positions_to_substitute"][batch_id]] = kwargs["states_to_substitute"][self.layer_idx][batch_id]
        return self.orig_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)

class ImplicitCoTModelWithRNN(nn.Module):
    """
    Implementation of Implicit Chain of Thought with RNN component
    Based on the second project's approach
    """
    def __init__(self, teacher_model_name, student_model_name, device, max_length=256):
        super().__init__()
        self.device = device

        # Save model names
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(device)
        for i in range(len(self.teacher_model.model.layers)):
            self.teacher_model.model.layers[i] = ModifiedDecoderLayer(self.teacher_model.model.layers[i], i)
        self.student_model = AutoModelForCausalLM.from_pretrained(student_model_name).to(device)
        for i in range(len(self.student_model.model.layers)):
            self.student_model.model.layers[i] = ModifiedDecoderLayer(self.student_model.model.layers[i], i)

        # Define hidden size
        self.teacher_hidden_size = self.teacher_model.config.hidden_size
        self.teacher_num_layers = self.teacher_model.config.num_hidden_layers
        self.student_hidden_size = self.student_model.config.hidden_size
        self.student_num_layers = self.student_model.config.num_hidden_layers

        # RNN component for implicit reasoning
        self.rnn = nn.LSTM(
            input_size=self.student_hidden_size,
            hidden_size=self.student_hidden_size,
            num_layers=1,
            batch_first=False,
            dropout=0,
            bidirectional=False
        )

        # Projection layers for attention mechanism
        self.key_proj = nn.Linear(self.student_hidden_size, self.student_hidden_size)
        self.query_proj = nn.Linear(self.student_hidden_size, self.student_hidden_size)
        self.out_proj = nn.Linear(self.student_hidden_size*2, self.student_hidden_size)

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*self.student_hidden_size, 4*self.student_hidden_size),
                nn.ReLU(),
                nn.Linear(4*self.student_hidden_size, self.student_hidden_size)
            ) for _ in range(self.teacher_num_layers)
        ])

        # Mixture component embeddings
        self.mixture_size = 1  # Default, can be configured
        self.mixture_components = nn.Embedding(self.mixture_size, self.student_hidden_size)

        # Temperature for softmax
        self.softmax_temperature = 0.05
        
        self.layer_norm = nn.LayerNorm(self.teacher_hidden_size, elementwise_affine=False)
        self.max_length = max_length
        
    def extract_states(self, input_ids):
        # Find the boundaries between input and CoT, and CoT and output
        first_sep_position = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=0)[0]
        second_sep_position = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=1)[0]

        # Forward the teacher to produce all hidden states
        outputs = self.teacher_model.forward(input_ids=input_ids, output_hidden_states=True)

        # Compute the positions to extract teacher states (t_l in the paper)
        layer_ids = torch.arange(start=0, end=self.teacher_num_layers).to(self.device)
        delta = (second_sep_position - first_sep_position) / (self.teacher_num_layers - 1)
        positions_to_extract = torch.round(first_sep_position + layer_ids * delta).clamp(max=second_sep_position)
        positions_to_extract_per_layer = positions_to_extract.unsqueeze(0)

        # Extract teacher states
        teacher_states_extracted = []
        for i, hidden_state in enumerate(outputs.hidden_states[:-1]):
            z = hidden_state.gather(1, positions_to_extract_per_layer[:,i].view(-1, 1, 1).expand(-1, -1, self.teacher_hidden_size)).squeeze(1)
            teacher_states_extracted.append(self.layer_norm(z))
        return teacher_states_extracted
    
    def teacher_forward(self, sample, with_answer=True):
        bos_tok, eos_tok = self.tokenizer.bos_token, self.tokenizer.eos_token
        prompt = f"{sample['query']} {bos_tok} {sample['reasoning'] if 'reasoning' in sample else sample['full_answer']} {eos_tok}"
        if with_answer:
            prompt += f" {sample['answer']} {eos_tok}"
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=self.max_length)
        labels = inputs['input_ids'].clone()
        outputs = self.teacher_model.forward(input_ids=inputs['input_ids'])

        mask = labels[...,1:].ge(0)
        correct_tokens = ((outputs.logits.argmax(-1)[...,:-1] == labels[...,1:]) * mask).sum()
        token_accuracy = correct_tokens / mask.sum()

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        del outputs, labels, mask, correct_tokens, shift_logits, shift_labels
        torch.cuda.empty_cache()
        return inputs, loss, token_accuracy
    
    def student_forward(self, sample):
        bos_tok, eos_tok = self.tokenizer.bos_token, self.tokenizer.eos_token
        prompt = f"{sample['query']} {bos_tok} {sample['answer']} {eos_tok}"
        inputs_nocot = self.tokenizer(prompt, return_tensors='pt', max_length=self.max_length)['input_ids']
        labels_nocot = inputs_nocot.clone()
        prompt = f"{sample['query']} {bos_tok} {sample['reasoning'] if 'reasoning' in sample else sample['full_answer']} {eos_tok}"
        inputs_ccot = self.tokenizer(prompt, return_tensors='pt', max_length=self.max_length)['input_ids']
        
        teacher_states = self.extract_states(inputs_ccot)
        sep_positions = get_sep_position(inputs_nocot, self.tokenizer.eos_token_id)
        teacher_states = [self.mlps[l](teacher_states[l]) for l in range(len(teacher_states))]

        # Forward while substituting teacher states
        outputs = self.student_model(
            input_ids=inputs_nocot, 
            positions_to_substitute=sep_positions, 
            states_to_substitute=teacher_states,
            requires_backward=False
        )

        labels_pred = outputs.logits.argmax(-1)
        mask = labels_nocot[...,1:].ge(0)
        correct_tokens = ((labels_pred[...,:-1] == labels_nocot[...,1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels_nocot[..., 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, token_accuracy
    
    def student_generate(self, sample, max_new_tokens=512, num_beams=1):
        bos_tok, eos_tok = self.tokenizer.bos_token, self.tokenizer.eos_token
        prompt = f"{sample['query']} {bos_tok} {sample['answer']} {eos_tok}"
        inputs_nocot = self.tokenizer(prompt, return_tensors='pt', max_length=self.max_length)['input_ids']
        sep_positions = get_sep_position(inputs_nocot, self.tokenizer.eos_token_id)
        teacher_states = [self.mlps[l](teacher_states[l]) for l in range(len(teacher_states))]
        inputs_nocot = inputs_nocot[:, :sep_positions + 1]
        beam_output = self.base_model.generate(
            input_ids=inputs_nocot,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            num_return_sequences=1,
            positions_to_substitute=sep_positions.repeat_interleave(num_beams, dim=0),
            states_to_substitute=[z[0:1].repeat_interleave(num_beams, dim=0) for z in teacher_states],
            mode='forward_student',
        )            
        return beam_output

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
            "hidden_size": self.student_hidden_size,
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

def extract_answer(text, eos_token):
    """Extract the final answer from text with eos_token delimiter"""
    possible_answers = text.split(eos_token)
    possible_answers = [a.strip() for a in possible_answers if len(a.strip()) > 0]
    return possible_answers[-1]

def get_sep_position(input_ids, sep_id, skip=0):
    batch_size = input_ids.shape[0]
    sep_positions = input_ids.new_zeros(batch_size).long()
    for batch_id in range(batch_size):
        mask = input_ids[batch_id].eq(sep_id)
        sep_position = mask.nonzero()[0, -1].item()
        for _ in range(skip):
            mask[sep_position] = False
            sep_position = mask.nonzero()[0, -1].item()
        sep_positions[batch_id] = sep_position
    return sep_positions

def run_sample_student(sample, tokenizer, icot_kd:ImplicitCoTModelWithRNN, max_length=256):
    bos_tok, eos_tok = tokenizer.bos_token, tokenizer.eos_token
    prompt = f"{sample['query']} {bos_tok} {sample['answer']} {eos_tok}"
    inputs_nocot = tokenizer(prompt, return_tensors='pt', max_length=max_length)['input_ids']
    labels_nocot = inputs_nocot.clone()
    prompt = f"{sample['query']} {bos_tok} {sample['reasoning'] if 'reasoning' in sample else sample['full_answer']} {eos_tok}"
    inputs_ccot = tokenizer(prompt, return_tensors='pt', max_length=max_length)['input_ids']
    teacher_states = icot_kd.extract_states(inputs_ccot)
    sep_positions = get_sep_position(inputs_nocot, tokenizer.eos_token_id)
    teacher_states = [icot_kd.mlps[l](teacher_states[l]) for l in range(len(teacher_states))]

    # Forward while substituting teacher states
    outputs = self.forward(input_ids, sep_positions, teacher_states)
    logits = outputs.logits

    labels_pred = logits.argmax(-1)
    mask = labels[...,1:].ge(0)
    correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
    total_tokens = mask.sum()
    token_accuracy = correct_tokens / total_tokens

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    outputs.loss = loss
    outputs.token_accuracy = token_accuracy
    outputs.total_correct = correct_tokens
    outputs.total_loss = loss * total_tokens
    outputs.total_tokens = total_tokens
    return outputs
    

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
    logger = Logger(
        log_dir=f"{output_path}/logs",
        experiment_name="icot_kd_training"
    )
    
    icot_kd = ImplicitCoTModelWithRNN(
        teacher_model_name=teacher_model_name,
        student_model_name=student_model_name,
        device=device
    ).to(device)
    icot_kd.teacher_model.train()
    
    logger.info("Training teacher model on CoT")
    teacher_opt = torch.optim.AdamW(list(icot_kd.teacher_model.parameters()), lr=5e-5)
    for step, sample in enumerate(train_dataset):
        teacher_opt.zero_grad()
        icot_kd.teacher_forward(sample)
        _, loss, tok_acc = icot_kd.teacher_forward(sample)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(icot_kd.teacher_model.parameters()), 1.0)
        teacher_opt.step()
        if step % 100 == 0:
            logger.info(f"Step {step}: loss = {loss} | token_acc = {tok_acc}")
        del loss, tok_acc
        torch.cuda.empty_cache()
    # evaluate 
    total_correct = 0
    with torch.no_grad():
        for step, sample in enumerate(eval_dataset):
            inputs, loss, tok_acc = icot_kd.teacher_forward(sample, with_answer=False)
            output = icot_kd.teacher_model.generate(input_ids=inputs['input_ids'], max_new_tokens=128)
            generated_text = icot_kd.tokenizer.decode(output[0])
            generated_answer = extract_answer(generated_text, icot_kd.tokenizer.eos_token)
            if generated_answer.strip() == sample["answer"].strip():
                total_correct += 1
            if step % 100 == 0:
                logger.info(f"Step {step}: loss = {loss} | token_acc = {tok_acc}")
            del inputs, loss, tok_acc, output
            torch.cuda.empty_cache()
    logger.info(f"Eval Accuracy = {total_correct / len(eval_dataset)}")
    del teacher_opt
    torch.cuda.empty_cache()
    
    icot_kd.teacher_model.eval()
    icot_kd.student_model.eval()
    for param in icot_kd.teacher_model.parameters():
        param.requires_grad = False
    student_opt = torch.optim.AdamW(list(icot_kd.student_model.parameters()), lr=5e-5)
    for epoch in range(5):
        for step, sample in enumerate(train_dataset):
            student_opt.zero_grad()
            loss, tok_acc = icot_kd.student_forward(sample)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(icot_kd.student_model.parameters()), 1.0)
            student_opt.step()
            if step % 100 == 0:
                logger.info(f"Step {step}: PPL = {loss.exp().item()} | token_acc = {tok_acc}")
        with torch.no_grad():
            for step, sample in enumerate(eval_dataset):
                loss, tok_acc = icot_kd.student_forward(sample)
                if step % 100 == 0:
                    logger.info(f"Step {step}: loss = {loss} | token_acc = {tok_acc}")
    # # Determine hidden size from the teacher model
    # hidden_size = teacher_model.config.hidden_size

    # # Initialize the implicit CoT model
    # icot_kd = ImplicitCoTModelWithRNN(
    #     teacher_model_name=teacher_model_name,
    #     student_model_name=student_model_name,
    #     hidden_size=hidden_size,
    #     device=device
    # ).to(device)

    # # Freeze teacher model parameters
    # for param in teacher_model.parameters():
    #     param.requires_grad = False

    # # Define optimizer
    # optimizer = optim.AdamW(icot_kd.parameters(), lr=learning_rate)

    # # Define loss function
    # loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # # Helper functions for prompt formatting
    # def format_teacher_prompt(query):
    #     """Format prompt for teacher to generate CoT reasoning"""
    #     if "mistral" in teacher_model_name.lower():
    #         return f"<s>[INST] Question: {query}\n Please solve this step-by-step. [/INST]"
    #     else:
    #         return f"<<SYS>>You are an expert in math word problems<</SYS>>\nQuestion: {query}\nPlease solve this step-by-step."

    # def format_student_prompt(query):
    #     """Format prompt for student to generate direct answer"""
    #     if "mistral" in student_model_name.lower():
    #         return f"<s>[INST] Question: {query}\n Generate the answer directly. Answer: [/INST]"
    #     else:
    #         return f"Question: {query}\n Generate the answer directly. Answer:"

    # # Generate teacher hidden states
    # def generate_teacher_states(query, answer):
    #     """Generate teacher hidden states for a given query"""
    #     # Format prompt with CoT reasoning
    #     prompt = format_teacher_prompt(query)

    #     # Tokenize
    #     inputs = tokenizer(
    #         prompt,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=512
    #     ).to(device)

    #     # Generate reasoning with teacher
    #     with torch.no_grad():
    #         outputs = teacher_model(
    #             input_ids=inputs.input_ids,
    #             attention_mask=inputs.attention_mask,
    #             output_hidden_states=True,
    #             return_dict=True
    #         )

    #         # Get hidden states from the last layer
    #         hidden_states = outputs.hidden_states[-1]

    #         # Get the last token's hidden state
    #         last_token_idx = inputs.attention_mask.sum(dim=1) - 1
    #         last_token_hidden = torch.stack([
    #             hidden_states[i, last_token_idx[i]]
    #             for i in range(inputs.input_ids.size(0))
    #         ])

    #         # Extract representative hidden states
    #         # For simplicity, we'll use the last token's hidden state
    #         teacher_state = last_token_hidden

    #     return teacher_state

    # # Training loop
    # best_val_loss = float('inf')

    # for epoch in range(num_epochs):
    #     # Training phase
    #     icot_kd.train()
    #     total_train_loss = 0

    #     # Process data in batches
    #     for i in range(0, len(train_dataset), batch_size):
    #         batch_data = train_dataset[i:i+batch_size]

    #         # Extract questions and answers
    #         questions = [item["query"] for item in batch_data]
    #         answers = [item["answer"] for item in batch_data]

    #         # Initialize batch loss
    #         batch_loss = 0

    #         # Process each question-answer pair in the batch
    #         for j, (question, answer) in enumerate(zip(questions, answers)):
    #             # Generate teacher states
    #             teacher_state = generate_teacher_states(question, answer)

    #             # Format student prompt
    #             prompt = format_student_prompt(question) + " "

    #             # Tokenize input
    #             inputs = icot_kd.student_tokenizer(
    #                 prompt,
    #                 return_tensors="pt",
    #                 padding=True,
    #                 truncation=True,
    #                 max_length=512
    #             ).to(device)

    #             # Tokenize target (prompt + answer)
    #             target_text = prompt  + answer
    #             target = icot_kd.student_tokenizer(
    #                 target_text,
    #                 return_tensors="pt",
    #                 padding=True,
    #                 truncation=True,
    #                 max_length=512
    #             ).to(device)

    #             input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
    #             # Forward pass through implicit CoT model
    #             while input_ids.shape[-1] < target.input_ids.shape[-1] - 1:
    #                 outputs = icot_kd(
    #                     input_ids=input_ids,
    #                     attention_mask=attention_mask,
    #                     teacher_states=teacher_state
    #                 )
    #                 input_ids = torch.cat([input_ids, outputs['logits'][0, -1].argmax().reshape(1,1)], dim=1)
    #                 attention_mask = torch.cat([attention_mask, torch.tensor([[1]]).to(device)], dim=1)
    #             outputs = icot_kd(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 teacher_states=teacher_state
    #             )
    #             # Compute loss
    #             logits = outputs["logits"]

    #             # Shift logits and labels for causal language modeling
    #             shift_logits = logits.contiguous()
    #             shift_labels = target.input_ids[:, 1:].contiguous()

    #             # Compute loss
    #             loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    #             # Accumulate loss
    #             batch_loss += loss

    #         # Average loss over batch
    #         batch_loss = batch_loss / len(batch_data)

    #         # Backward pass
    #         optimizer.zero_grad()
    #         batch_loss.backward()
    #         optimizer.step()

    #         # Accumulate total loss
    #         total_train_loss += batch_loss.item()

    #     avg_train_loss = total_train_loss / (len(train_dataset) / batch_size)
    #     print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

    #     # Validation phase
    #     icot_kd.eval()
    #     total_val_loss = 0

    #     with torch.no_grad():
    #         for i in range(0, len(eval_dataset), batch_size):
    #             batch_data = eval_dataset[i:i+batch_size]

    #             # Extract questions and answers
    #             questions = [item["query"] for item in batch_data]
    #             answers = [item["answer"] for item in batch_data]

    #             # Initialize batch loss
    #             batch_loss = 0

    #             # Process each question-answer pair in the batch
    #             for j, (question, answer) in enumerate(zip(questions, answers)):
    #                 # Generate teacher states
    #                 teacher_state = generate_teacher_states(question, answer)

    #                 # Format student prompt
    #                 prompt = format_student_prompt(question)

    #                 # Tokenize input
    #                 inputs = icot_kd.student_tokenizer(
    #                     prompt,
    #                     return_tensors="pt",
    #                     padding=True,
    #                     truncation=True,
    #                     max_length=512
    #                 ).to(device)

    #                 # Tokenize target (prompt + answer)
    #                 target_text = prompt + " " + answer
    #                 target = icot_kd.student_tokenizer(
    #                     target_text,
    #                     return_tensors="pt",
    #                     padding=True,
    #                     truncation=True,
    #                     max_length=512
    #                 ).to(device)
                    
    #                 input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
    #                 # Forward pass through implicit CoT model
    #                 while input_ids.shape[-1] < target.input_ids.shape[-1] - 1:
    #                     outputs = icot_kd(
    #                         input_ids=input_ids,
    #                         attention_mask=attention_mask,
    #                         teacher_states=teacher_state
    #                     )
    #                     input_ids = torch.cat([input_ids, outputs['logits'][0, -1].argmax().reshape(1,1)], dim=1)
    #                     attention_mask = torch.cat([attention_mask, torch.tensor([[1]]).to(device)], dim=1)
    #                 outputs = icot_kd(
    #                     input_ids=input_ids,
    #                     attention_mask=attention_mask,
    #                     teacher_states=teacher_state
    #                 )

    #                 # Compute loss
    #                 logits = outputs["logits"]

    #                 # Shift logits and labels for causal language modeling
    #                 shift_logits = logits.contiguous()
    #                 shift_labels = target.input_ids[:, 1:].contiguous()

    #                 # Compute loss
    #                 loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    #                 # Accumulate loss
    #                 batch_loss += loss

    #             # Average loss over batch
    #             batch_loss = batch_loss / len(batch_data)

    #             # Accumulate total loss
    #             total_val_loss += batch_loss.item()

    #     avg_val_loss = total_val_loss / (len(eval_dataset) / batch_size)
    #     print(f"Validation Loss: {avg_val_loss:.4f}")

    #     # Save best model
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         icot_kd.save_pretrained(output_path)
    #         print(f"Saved best model with validation loss: {best_val_loss:.4f}")

    # # Clean up to save memory
    # del teacher_model
    # gc.collect()
    # torch.cuda.empty_cache()

    # return icot_kd