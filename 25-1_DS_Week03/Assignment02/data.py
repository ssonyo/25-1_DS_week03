
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import Counter
from typing import List, Tuple, Union


class WordSequenceDataset(Dataset):
    def __init__(self, data: Union[str, List[str]], window_size: int = 2, max_vocab_size: int = 100):
        self.window_size = window_size
        self.samples = []

        if isinstance(data, str):  
            with open(data, "r", encoding="utf-8") as f:
                lines = [line.strip().lower() for line in f.readlines() if line.strip()]
        else:  
            lines = [line.strip().lower() for line in data if line.strip()]

        self._build_vocab(lines, max_vocab_size)
        self._build_samples(lines)

    def _build_vocab(self, lines, max_vocab_size):
        words = []
        for line in lines:
            tokens = line.split()
            words.extend(tokens)

        freq = Counter(words)
        most_common = freq.most_common(max_vocab_size - 2)  # reserve 0:<pad>, 1:<unk>
        self.token2idx = {"<pad>": 0, "<unk>": 1}
        for word, _ in most_common:
            self.token2idx[word] = len(self.token2idx)
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)

    def _build_samples(self, lines):
        for line in lines:
            tokens = line.split()
            if len(tokens) < self.window_size + 1:
                continue

            for i in range(len(tokens) - self.window_size):
                seq = tokens[i:i + self.window_size]
                target = tokens[i + self.window_size]

                seq_ids = [self.token2idx.get(tok, 1) for tok in seq]
                target_id = self.token2idx.get(target, 1)

                if 1 in seq_ids or target_id == 1:
                    continue

                self.samples.append((torch.tensor(seq_ids), torch.tensor(target_id)))

        print(f"[DEBUG] final samples : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
