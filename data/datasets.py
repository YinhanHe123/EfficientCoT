from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

def load_gsm8k_dataset(data_path=None):
    """Load and prepare the GSM8K dataset"""
    # Load from HuggingFace datasets or from local path
    gsm8k = load_dataset(data_path, 'main')
    train_dataset = gsm8k['train']
    eval_dataset = gsm8k['test']
    # Create custom PyTorch datasets
    train_data = GSM8KDataset(train_dataset)
    eval_data = GSM8KDataset(eval_dataset)

    return train_data, eval_data

class GSM8KDataset(Dataset):
    """Dataset class for GSM8K problems"""
    def __init__(self, hf_dataset, max_length=1024):
        self.dataset = list(hf_dataset)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Extract question and answer from the dataset
        question = item['question']

        # GSM8K format has answers with reasoning steps followed by the final answer
        full_answer = item['answer']

        condensed_reasoning = item['condensed_reasoning'] if 'condensed_reasoning' in item else None

        # Extract reasoning and final answer
        reasoning_parts = full_answer.split('####')
        reasoning = reasoning_parts[0].strip()

        # The final answer comes after ####
        final_answer = reasoning_parts[1].strip() if len(reasoning_parts) > 1 else ""

        return {
            "query": question,
            "reasoning": reasoning,
            "answer": final_answer,
            "full_answer": full_answer,
            "condensed_reasoning": condensed_reasoning
        }

    def update_item(self, idx, key, value):
        """Add or update a field in the dataset at the specified index"""
        # If we're adding a field that's one of our transformed fields
        if key in ["query", "reasoning", "answer", "full_answer"]:
            # Need to map back to the original dataset format
            if key == "query":
                self.dataset[idx]["question"] = value
            elif key == "reasoning":
                # Update reasoning while preserving the answer part
                current_answer = self.dataset[idx]["answer"]
                reasoning_parts = current_answer.split('####')
                if len(reasoning_parts) > 1:
                    self.dataset[idx]["answer"] = value + "\n#### " + reasoning_parts[1].strip()
                else:
                    self.dataset[idx]["answer"] = value
            elif key == "answer":
                # Update just the answer part after ####
                current_answer = self.dataset[idx]["answer"]
                reasoning_parts = current_answer.split('####')
                if len(reasoning_parts) > 1:
                    self.dataset[idx]["answer"] = reasoning_parts[0].strip() + "\n#### " + value
                else:
                    self.dataset[idx]["answer"] = current_answer + "\n#### " + value
            elif key == "full_answer":
                self.dataset[idx]["answer"] = value
        else:
            # For new fields, we need to add them to the underlying dataset
            # Note: This will only work if the underlying dataset supports item assignment
            self.dataset[idx][key] = value

def create_dataloaders(train_dataset, eval_dataset, batch_size=16):
    """Create PyTorch DataLoaders for training and evaluation"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, eval_loader