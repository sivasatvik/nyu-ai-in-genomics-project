from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import torch
from torch.utils.data import Dataset


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYXBZUO"
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


@dataclass
class SequenceTokenizer:
    vocab: str = AA_VOCAB

    def __post_init__(self):
        self.itos = SPECIAL_TOKENS + list(self.vocab)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, seq: str, max_len: int) -> List[int]:
        seq_ids = [self.stoi.get(ch, self.stoi["X"]) for ch in seq[: max_len - 2]]
        ids = [self.bos_id] + seq_ids + [self.eos_id]
        ids += [self.pad_id] * (max_len - len(ids))
        return ids


class EnvSequenceDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer: SequenceTokenizer, max_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        seq_ids = torch.tensor(self.tokenizer.encode(r["sequence"], self.max_len), dtype=torch.long)
        env = torch.tensor(r["env_vector"], dtype=torch.float32)
        fn_label = torch.tensor(r["function_label"], dtype=torch.long)

        x = seq_ids[:-1]
        y = seq_ids[1:]
        mask = x.ne(self.tokenizer.pad_id)
        return {"env": env, "x": x, "y": y, "mask": mask, "fn_label": fn_label}
