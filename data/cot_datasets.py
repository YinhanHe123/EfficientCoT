from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from data.gpt4pair import ReasoningPairsGenerator
from tqdm import tqdm
import os

def load_raw_dataset(data_path=None):
    """Load and prepare the raw dataset"""
    # Load from HuggingFace datasets or from local path
    data_name = data_path.split('/')[-1]
    if 'gsm8k' in data_path:
        raw = load_dataset(data_path, 'main')
    elif 'SVAMP' in data_path:
        raw = load_dataset(data_path, 'default')
    elif 'MultiArith' in data_path:
        processed_path = '/data/nee7ne/huggingface/datasets/processed_MultiArith'
        if os.path.exists(processed_path):
            raw = load_from_disk(processed_path)
        else:
            raw = load_dataset("ChilleD/MultiArith", 'default')
            generator = ReasoningPairsGenerator()
            def generate_mtarf_reasoning(example):
                # Generate reasoning using GPT-4o-mini
                query = example['question']
                reasoning = generator.generate_reasoning(query)
                example['answer'] = reasoning
                example['final_answer'] = example['final_ans']
                return example
            raw = raw.map(generate_mtarf_reasoning)
            os.makedirs(processed_path, exist_ok=True)
            raw.save_to_disk(processed_path)

        # raw = load_dataset(data_path, 'default')
        # # generate reasoning with gpt 4o-mini
        # generator = ReasoningPairsGenerator()
        # for i in tqdm(range(len(raw['train']))):
        #     query = raw['train'][i]['question']
        #     reasoning = generator.generate_reasoning(query)
        #     raw['train'][i]['answer'] = reasoning
        #     raw['train'][i]['final_answer'] = raw['train'][i]['final_ans']
        # for i in tqdm(range(len(raw['test']))):
        #     query = raw['train'][i]['question']
        #     reasoning = generator.generate_reasoning(query)
        #     raw['train'][i]['answer'] = reasoning
        #     raw['train'][i]['final_answer'] = raw['train'][i]['final_ans']
        # # # save the dataset to the path
        # raw['train'].save_to_disk(data_path + '/train')
        # raw['test'].save_to_disk(data_path + '/test')


    train_dataset = raw['train'].select(range(400))  # For debugging
    eval_dataset = raw['test'].select(range(100))  # For debugging
    # Create custom PyTorch datasets

    train_data = RawDataset(train_dataset, data_name)
    eval_data = RawDataset(eval_dataset, data_name)
    return train_data, eval_data

class RawDataset(Dataset):
    """Dataset class for reasoning tasks"""
    def __init__(self, hf_dataset, name, max_length=1024):
        self.dataset = list(hf_dataset)
        self.max_length = max_length
        self.name = name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(idx.start, idx.stop)] 
        item = self.dataset[idx]

        if self.name == 'gsm8k':
            # Extract question and answer from the dataset
            question = item['question']
            # raw format has answers with reasoning steps followed by the final answer
            full_answer = item['answer']
            condensed_reasoning = item['condensed_reasoning'] if 'condensed_reasoning' in item else None
            # Extract reasoning and final answer
            reasoning_parts = full_answer.split('####')
            reasoning = reasoning_parts[0].strip()
            # The final answer comes after ####
            final_answer = reasoning_parts[1].strip() if len(reasoning_parts) > 1 else ""
        elif self.name == 'SVAMP':
            question = item['question_concat']
            reasoning = 'The question is in type of ' + item['Type'] + ' and we solve it by calculating ' + item['Equation']
            full_answer = reasoning + '####' + item['Answer']
            condensed_reasoning = item['condensed_reasoning'] if 'condensed_reasoning' in item else None
            final_answer = item['Answer']
        elif self.name == 'MultiArith': # gt reasoning not available.
            question = item['question']
            reasoning = ""
            full_answer = item['answer']
            condensed_reasoning = item['condensed_reasoning'] if 'condensed_reasoning' in item else None
            final_answer = item['final_answer']
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
        num_workers=3
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3
    )

    return train_loader, eval_loader