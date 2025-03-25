import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SimCSEDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_len=32):
        self.sentences = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        with open(file_path, 'r') as file:
            for line in file:
                self.sentences.append(line.strip())

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)
