"""
Dataset downloading, tokenization, caching, and DataLoader factory.
Ensures tokenization runs exactly once and restricts memory usage.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from config import CACHE_DIR, MODEL_NAME, MAX_TRAIN_TOKENS, MAX_VAL_TOKENS, BLOCK_SIZE, BATCH_SIZE

class TokenizedDataset(Dataset):
    def __init__(self, data, block_size):
        self.block_size = block_size
        # Use non-overlapping chunks to keep dataset size manageable
        n_chunks = (len(data) - 1) // block_size
        self.data = data[:n_chunks * block_size + 1]
        self.n_chunks = n_chunks

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, i):
        start = i * self.block_size
        x = self.data[start: start + self.block_size]
        y = self.data[start + 1: start + self.block_size + 1]
        return x, y

def get_dataloaders():
    """Returns train and validation dataloaders for the WikiText-103 subset."""
    cache_train = os.path.join(CACHE_DIR, "train.pt")
    cache_val = os.path.join(CACHE_DIR, "val.pt")

    if os.path.exists(cache_train) and os.path.exists(cache_val):
        print("Loading tokenized dataset from cache...")
        train_data = torch.load(cache_train)
        val_data = torch.load(cache_val)
    else:
        print("Downloading and tokenizing WikiText-103 (streaming)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        dataset_val = load_dataset("wikitext", "wikitext-103-v1", split="test", streaming=True)

        def _tokenize_split(split_generator, max_tokens):
            tokens = []
            for item in split_generator:
                if len(item["text"].strip()) == 0:
                    continue
                toks = tokenizer(item["text"])["input_ids"]
                tokens.extend(toks)
                if len(tokens) >= max_tokens:
                    break
            return torch.tensor(tokens[:max_tokens], dtype=torch.long)

        train_data = _tokenize_split(dataset, MAX_TRAIN_TOKENS)
        val_data = _tokenize_split(dataset_val, MAX_VAL_TOKENS)

        torch.save(train_data, cache_train)
        torch.save(val_data, cache_val)
        print("Tokenization complete and cached.")

    train_ds = TokenizedDataset(train_data, BLOCK_SIZE)
    val_ds = TokenizedDataset(val_data, BLOCK_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader
