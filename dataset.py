"""Dataset class for text data."""

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Simple text dataset for causal language modeling."""
    
    def __init__(
        self, 
        filepath: str, 
        tokenizer, 
        block_size: int = 1024,
        token_offset: int = 0,      # NEW: Skip first N tokens
        max_tokens: int = None      # NEW: Limit total tokens
    ):
        """
        Args:
            filepath: Path to text file
            tokenizer: HuggingFace tokenizer
            block_size: Sequence length for training
            token_offset: Number of tokens to skip from start
            max_tokens: Maximum tokens to use (None = use all)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Tokenize full text
        all_tokens = self.tokenizer.encode(text)
        
        # Apply offset (skip already-trained tokens)
        if token_offset > 0:
            all_tokens = all_tokens[token_offset:]
            print(f"Skipping first {token_offset:,} tokens")
        
        # Apply max limit if specified
        if max_tokens is not None:
            all_tokens = all_tokens[:max_tokens]
        
        self.tokens = all_tokens
        
        print(f"Dataset loaded: {len(self.tokens):,} tokens, {len(self):,} samples")
        print(f"Token range: [{token_offset:,} : {token_offset + len(self.tokens):,}]")

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx):
        input_ids = self.tokens[idx:idx + self.block_size]
        target_ids = self.tokens[idx + 1:idx + self.block_size + 1]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(target_ids, dtype=torch.long)
        }


def calculate_tokens_trained(steps: int, batch_size: int, block_size: int) -> int:
    """Calculate total tokens trained given steps, batch size, and block size."""
    return steps * batch_size * block_size