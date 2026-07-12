"""
Dataset
========
PyTorch Dataset that serves tokenised beat sequences for training.
Supports both pattern-level and arrangement-level training.
"""

import logging
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BeatDataset(Dataset):
    """
    Dataset of tokenised beat sequences.

    Two modes:
      - "pattern": Each sample is a single pattern sequence
      - "arrangement": Each sample is a full arrangement sequence
      - "both": Interleaves patterns and arrangements

    Sequences are padded/truncated to max_seq_len.
    """

    def __init__(
        self,
        data_path: str | Path,
        max_seq_len: int = 2048,
        mode: str = "both",
        pad_token_id: int = 0,
    ):
        data_path = Path(data_path)

        with open(data_path, "rb") as f:
            all_tokenized = pickle.load(f)

        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.sequences: list[list[int]] = []

        for project in all_tokenized:
            if mode in ("pattern", "both"):
                for pat_seq in project["pattern_sequences"]:
                    if len(pat_seq) >= 5:  # Skip tiny patterns
                        self.sequences.append(pat_seq)

            if mode in ("arrangement", "both"):
                arr_seq = project["arrangement_sequence"]
                if len(arr_seq) >= 10:
                    # For long arrangements, split into overlapping chunks
                    if len(arr_seq) <= max_seq_len:
                        self.sequences.append(arr_seq)
                    else:
                        stride = max_seq_len // 2
                        for start in range(0, len(arr_seq) - max_seq_len // 4, stride):
                            chunk = arr_seq[start : start + max_seq_len]
                            if len(chunk) >= max_seq_len // 4:
                                self.sequences.append(chunk)

        logger.info(f"BeatDataset: {len(self.sequences)} sequences (mode={mode})")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]

        # Truncate to max_seq_len (need +1 for input/target shift)
        if len(seq) > self.max_seq_len + 1:
            # Random crop for data augmentation
            start = random.randint(0, len(seq) - self.max_seq_len - 1)
            seq = seq[start : start + self.max_seq_len + 1]

        # Pad if shorter
        if len(seq) < self.max_seq_len + 1:
            pad_len = self.max_seq_len + 1 - len(seq)
            seq = seq + [self.pad_token_id] * pad_len

        seq_tensor = torch.tensor(seq, dtype=torch.long)

        # Input: all tokens except last
        # Target: all tokens except first (shifted by 1)
        input_ids = seq_tensor[:-1]
        target_ids = seq_tensor[1:]

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
        }


def create_data_splits(
    data_path: str | Path,
    max_seq_len: int = 2048,
    mode: str = "both",
    eval_split: float = 0.1,
    seed: int = 42,
    pad_token_id: int = 0,
) -> tuple[Dataset, Dataset]:
    """Create train/val datasets from the tokenized data."""
    full_dataset = BeatDataset(data_path, max_seq_len, mode, pad_token_id)

    n_total = len(full_dataset)
    n_val = max(1, int(n_total * eval_split))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    logger.info(f"Split: {n_train} train, {n_val} val")
    return train_dataset, val_dataset
