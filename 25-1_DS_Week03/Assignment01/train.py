import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import torch.nn.functional as F
from model import SimCSEModel
from dataset import SimCSEDataset

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 32
CHECKPOINT_PATH = './checkpoint'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = SimCSEDataset('data/simple_corpus.txt', tokenizer, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SimCSEModel(MODEL_NAME).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids, attention_mask = batch
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        optimizer.zero_grad()
        embeddings = model(input_ids, attention_mask=attention_mask)

        # Contrastive loss: maximize similarity for the same sentence
        loss = criterion(embeddings, embeddings)  # This is a placeholder
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f"model_epoch_{epoch + 1}.pth"))
