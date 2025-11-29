"""Dataset class for text data."""

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Simple text dataset for causal language modeling."""
    
    def __init__(self, filepath: str, tokenizer, block_size: int = 1024):
        """
        Args:
            filepath: Path to text file
            tokenizer: HuggingFace tokenizer
            block_size: Sequence length for training
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.tokens = self.tokenizer.encode(text)
        
        print(f"Dataset loaded: {len(self.tokens):,} tokens, {len(self):,} samples")

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        input_ids = self.tokens[idx:idx + self.block_size]
        target_ids = self.tokens[idx + 1:idx + self.block_size + 1]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(target_ids, dtype=torch.long)
        }