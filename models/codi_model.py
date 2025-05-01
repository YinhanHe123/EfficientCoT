import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import os
import time

class CODIModel(nn.Module):
    """
    Implementation of Continuous Chain-of-Thought via Self-Distillation (CODI) model
    from the paper "CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation"
    by Zhenyi Shen et al.
    """

    def __init__(self, base_model_name, num_continuous_tokens=6, device="cuda"):
        """
        Initialize the CODI model.

        Args:
            base_model_name: Name of the base LLM model
            num_continuous_tokens: Number of continuous thought tokens to use
            device: Device to run the model on (cuda or cpu)
        """
        super().__init__()
        self.device = device
        self.base_model_name = base_model_name
        self.num_continuous_tokens = num_continuous_tokens

        # Load the base model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add special tokens for CODI if they don't exist
        special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)

        # Resize token embeddings if new tokens were added
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                std = self.model.get_input_embeddings().weight.std().item()
                for token in ["<bot>", "<eot>"]:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    self.model.get_input_embeddings().weight[token_id].normal_(mean=0.0, std=std)

        self.bot_token_id = torch.tensor(self.tokenizer.convert_tokens_to_ids("<bot>")).reshape(1,1)
        self.eot_token_id = torch.tensor(self.tokenizer.convert_tokens_to_ids("<eot>")).reshape(1,1)
        self.eos_token_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)).reshape(1,1)
        self.answer_prompt = self.tokenizer("The answer is: ", return_tensors="pt", add_special_tokens=False)["input_ids"]

        # Projection layer for continuous thoughts
        self.projection_layer = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
        ).to(device)

        # Loss functions
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
        self.smooth_l1 = torch.nn.SmoothL1Loss()

    def forward_student(
        self, query_inputs: torch.Tensor, answer_inputs: torch.Tensor = None
    ):
        """
        Student forward pass: Generate continuous thoughts autoregressively

        Args:
            query_inputs: query token IDs
            answer_inputs: answer token IDs
        Returns:
            Continuous thought hidden states, student model outputs, and student loss
        """
        # query_embeds = self.model.get_input_embeddings()(query_inputs)
        # bot_token_embeds = self.model.get_input_embeddings()(self.bot_token_id.to(self.device))
        # bot_token_embeds.requires_grad = True
        # # Concatenate the embeddings
        # input_embeds = torch.cat([query_embeds, bot_token_embeds], dim=1)

        # # Run the model with embeddings input instead of IDs
        # student_outputs = self.model(
        #     inputs_embeds=input_embeds,
        #     output_hidden_states=True
        # )
        student_input_ids = torch.cat([query_inputs, self.bot_token_id.to(self.device)], dim=1)
        student_outputs = self.model(input_ids=student_input_ids, output_hidden_states=True)
        continuous_tokens = []
        latent = self.projection_layer(student_outputs["hidden_states"][-1][:, -1].unsqueeze(1))
        continuous_tokens.append(latent.squeeze(0))
        past_key_values = student_outputs["past_key_values"]
        for i in range(self.num_continuous_tokens - 1):
            student_outputs = self.model(
                inputs_embeds=latent,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            latent = self.projection_layer(student_outputs["hidden_states"][-1])
            continuous_tokens.append(latent.squeeze(0))
            past_key_values = student_outputs["past_key_values"]

        continuous_tokens = torch.stack(continuous_tokens, dim=1)
        student_input_ids = torch.cat([self.eot_token_id.to(self.device), self.answer_prompt.to(self.device)], dim=-1)
        if answer_inputs is not None:
            student_input_ids = torch.cat([student_input_ids, answer_inputs], dim=-1)

        # eot_token_embeds = self.model.get_input_embeddings()(self.eot_token_id.to(self.device))
        # eot_token_embeds.requires_grad = True
        # answer_prompt_embeds = self.model.get_input_embeddings()(self.answer_prompt.to(self.device))
        # end_embs = torch.cat([eot_token_embeds, answer_prompt_embeds], dim=1)
        # if answer_inputs is not None:
        #     end_embs = torch.cat([end_embs, self.model.get_input_embeddings()(answer_inputs)], dim=1)

        # end_embeds = torch.cat([eot_token_embeds, answer_prompt_embeds], dim=1)

        student_outputs = self.model(
            input_ids=student_input_ids,
            # inputs_embeds=end_embs,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )

        student_loss = None
        if answer_inputs is not None:
            # labels = torch.cat([answer_inputs, self.eos_token_id.to(self.device)], dim=1).reshape(-1)
            labels = answer_inputs.reshape(-1)
            student_loss = self.cross_entropy(student_outputs["logits"][:, self.eot_token_id.shape[1] + self.answer_prompt.shape[1] - 1:-1].squeeze(0), labels).mean()
        del latent, past_key_values, student_input_ids
        return student_outputs, student_loss, continuous_tokens

    def forward_teacher(
        self,
        query_inputs: torch.Tensor,
        cot_inputs: torch.Tensor = None,
        answer_inputs: torch.Tensor = None,
    ):
        """
        Teacher forward pass: Process query with the ground-truth CoT

        Args:
            query_inputs: query token IDs
            cot_inputs: reasoning token IDs
            answer_inputs: answer token IDs
        Returns:
            teacher outputs and teacher loss
        """
        teacher_input_ids = torch.cat([query_inputs, self.answer_prompt.to(self.device)], dim=-1)
        if cot_inputs is not None:
            teacher_input_ids = torch.cat([query_inputs, cot_inputs, self.answer_prompt.to(self.device), answer_inputs], dim=1)
        teacher_outputs = self.model(input_ids=teacher_input_ids, output_hidden_states=True)

        teacher_loss = None
        if cot_inputs is not None:
            labels = torch.cat([cot_inputs, self.answer_prompt.to(self.device), answer_inputs, self.eos_token_id.to(self.device)], dim=1).reshape(-1)
            teacher_loss = self.cross_entropy(teacher_outputs["logits"][:, query_inputs.shape[1] - 1:].squeeze(0), labels).mean()
            del labels
        del teacher_input_ids
        return teacher_outputs, teacher_loss

    def forward(self, data: dict[str, str]):
        """
        Combined forward pass for training with self-distillation

        Args:
            data: dataset sample that includes the query, reasoning, and answer
        Returns:
            Loss from self-distillation, student loss, and teacher loss
        """
        query_inputs = self.tokenizer(
            data["query"], return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

        if "reasoning" in data and len(data["reasoning"]) > 0:
            steps = data["reasoning"].split("\n")
            cot = "\n".join(steps[:-1] if len(steps) > 1 else steps)
        elif "full_answer" in data and len(data["full_answer"]) > 0:
            steps = data["full_answer"].split("### Conclusion")
            cot = steps[0]
            if len(steps) == 1:
                steps = data["full_answer"].split("\n\n")
                cot = "\n\n".join(steps[:-1])

        cot_inputs = self.tokenizer(
           cot,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)
        answer_inputs = self.tokenizer(
            data["answer"], return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

        teacher_outputs, teacher_loss = self.forward_teacher(
            query_inputs, cot_inputs, answer_inputs
        )
        student_outputs, student_loss, _ = self.forward_student(
            query_inputs, answer_inputs
        )

        distill_loss = self.smooth_l1(
            teacher_outputs["logits"][
                :, query_inputs.shape[1] + cot_inputs.shape[1] + self.answer_prompt.shape[1] - 1 :
            ],
            student_outputs["logits"][
                :, self.eot_token_id.shape[1] + self.answer_prompt.shape[1] - 1 :
            ],
        )

        del query_inputs, cot_inputs, answer_inputs
        torch.cuda.empty_cache()
        return teacher_loss, student_loss, distill_loss

    def generate(self, data, max_new_tokens=30, temperature=0.7, top_p=0.9, do_sample=True):
        """
        Generate an answer using the continuous thoughts

        Args:
            data: dataset sample that includes the query
            max_new_tokens: the number of tokens to generate
            temperature: text generation temperature
            top_p: text generation top_p
        Returns:
            Generated answer
        """
        tokenize_begin = time.time() # DEBUG
        query_inputs = self.tokenizer(data["query"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)

        token_end = time.time()      # DEBUG
        print(f"Tokenization time: {token_end - tokenize_begin:.2f} seconds") # DEBUG
        student_begin = time.time() # DEBUG
        _, _, continuous_hidden = self.forward_student(query_inputs)
        student_end = time.time()   # DEBUG
        print(f"Student forward time: {student_end - student_begin:.2f} seconds") # DEBUG

        # Get embeddings directly
        query_embeds = self.model.get_input_embeddings()(query_inputs)
        bot_token_embeds = self.model.get_input_embeddings()(self.bot_token_id.to(self.device))
        eot_token_embeds = self.model.get_input_embeddings()(self.eot_token_id.to(self.device))
        answer_prompt_embeds = self.model.get_input_embeddings()(self.answer_prompt.to(self.device))

        # Concatenate for input
        embed_begin = torch.cat([query_embeds, bot_token_embeds], dim=1)
        embed_end = torch.cat([eot_token_embeds, answer_prompt_embeds], dim=1)
        input_embeds = torch.cat([embed_begin, continuous_hidden, embed_end], dim=1)
        input_embeds = torch.cat([query_embeds, answer_prompt_embeds, bot_token_embeds, continuous_hidden, eot_token_embeds], dim=1) # DEBUG

        # input_begin = torch.cat([query_inputs, self.bot_token_id.to(self.device)], dim=1)
        # input_end = torch.cat([self.eot_token_id.to(self.device), self.answer_prompt.to(self.device)], dim=-1,)

        # embed_begin = self.model.get_input_embeddings()(input_begin)
        # embed_end = self.model.get_input_embeddings()(input_end)
        # input_embeds = torch.cat([embed_begin, continuous_hidden, embed_end], dim=1)

        generate_begin = time.time() # DEBUG
        outputs = self.model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=max_new_tokens + input_embeds.shape[1],
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generate_end = time.time() # DEBUG
        print(f"Generation time: {generate_end - generate_begin:.2f} seconds")

        tokenize_decde_begin = time.time() # DEBUG
        answer = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tokenize_decode_end = time.time() # DEBUG
        print(f"Tokenization and decoding time: {tokenize_decode_end - tokenize_decde_begin:.2f} seconds") # DEBUG
        # del query_inputs, input_begin, input_end, embed_begin, embed_end, input_embeds, outputs
        del query_inputs, embed_begin, embed_end, input_embeds, outputs
        torch.cuda.empty_cache()
        return answer

    def apply_lora(self, rank=128, alpha=32):
        """
        Apply LoRA to the model for efficient fine-tuning

        Args:
            rank: Rank for LoRA adaptation
            alpha: Alpha parameter for LoRA

        Returns:
            Model with LoRA applied
        """
        # Define target modules (attention layers)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
        )

        # Enable input require grads
        self.model.enable_input_require_grads()

        for name, param in self.model.named_parameters():
            if "embed_tokens" in name:
                param.requires_grad = True

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        return self

    @classmethod
    def from_pretrained(cls, path, device):
        """
        Load a pretrained CODI model

        Args:
            path: Path to the saved model

        Returns:
            Loaded CODI model
        """
        # Load config
        config_path = os.path.join(path, "config.pt")
        config = torch.load(config_path)

        # Initialize model with loaded config
        model = cls(
            config["base_model_name"],
            num_continuous_tokens=config["num_continuous_tokens"],
            device=device,
        )

        # Load state dict
        model_path = os.path.join(path, "model.pt")
        model.load_state_dict(torch.load(model_path, map_location=device if isinstance(device, str) else f"cuda:{device}"))

        return model

    def save_pretrained(self, path):
        """
        Save the CODI model

        Args:
            path: Path to save the model
        """
        os.makedirs(path, exist_ok=True)

        # Save config
        config = {
            "base_model_name": self.base_model_name,
            "num_continuous_tokens": self.num_continuous_tokens,
            "device": self.device,
        }
        torch.save(config, os.path.join(path, "config.pt"))

        # Save state dict
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
