import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SimCSEDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_len=32):
        self.sentences = [s.strip() for s in sentences if s.strip() != ""]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoded = self.tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

