from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from transformers import pipeline

class ReasoningPairsGenerator:
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto' # Automatically distribute across GPUs
                         # Memory limits per GPU
                        )
        self.device = device

    def generate_reasoning(self, query, max_length=1024):
        """Generate detailed reasoning for a given query"""
        prompt = f"Question: {query}\nLet's solve this step-by-step:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        # pipe = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
        # reasoning = pipe(prompt, max_length=max_length)[0]['generated_text']

        reasoning = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the reasoning part (remove the prompt)
        reasoning = reasoning.replace(prompt, "").strip()
        return reasoning

    def generate_condensed_reasoning(self, original_reasoning, max_length=1024):
        """Generate condensed version of the original reasoning"""
        prompt = f"Original reasoning: {original_reasoning}\n\nCondense the above reasoning into a VERY VERY concise version:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        condensed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the condensed part (remove the prompt)
        condensed = condensed.replace(prompt, "").strip()

        return condensed

    def create_reasoning_pair(self, query):
        """Generate a pair of original and condensed reasoning for a query"""
        original_reasoning = self.generate_reasoning(query)
        condensed_reasoning = self.generate_condensed_reasoning(original_reasoning)

        return {
            "query": query,
            "original_reasoning": original_reasoning,
            "condensed_reasoning": condensed_reasoning
        }

    def create_dataset(self, queries):
        """Create a dataset of reasoning pairs from a list of queries"""
        dataset = []

        for i in tqdm(range(len(queries))):
            pair = self.create_reasoning_pair(queries[i])
            dataset.append(pair)

        return dataset