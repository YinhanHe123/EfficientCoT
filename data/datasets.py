from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

def load_gsm8k_dataset(data_path=None):
    """Load and prepare the GSM8K dataset"""
    # Load from HuggingFace datasets or from local path
    gsm8k = load_dataset(data_path, 'main')
    train_dataset = gsm8k['train'].select(range(4)) # [:4]  is for testing
    eval_dataset = gsm8k['test'].select([0])  # [0] is for testing
    # Create custom PyTorch datasets
    train_data = GSM8KDataset(train_dataset)
    eval_data = GSM8KDataset(eval_dataset)

    return train_data, eval_data

class GSM8KDataset(Dataset):
    """Dataset class for GSM8K problems"""
    def __init__(self, hf_dataset, max_length=1024):
        self.dataset = hf_dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Extract question and answer from the dataset
        question = item['question']

        # GSM8K format has answers with reasoning steps followed by the final answer
        full_answer = item['answer']

        # Extract reasoning and final answer
        reasoning_parts = full_answer.split('####')
        reasoning = reasoning_parts[0].strip()

        # The final answer comes after ####
        final_answer = reasoning_parts[1].strip() if len(reasoning_parts) > 1 else ""

        return {
            "query": question,
            "reasoning": reasoning,
            "answer": final_answer,
            "full_answer": full_answer
        }

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