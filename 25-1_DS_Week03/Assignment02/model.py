
import torch
import torch.nn as nn
from embeddings import TokenEmbedding, PositionEmbedding
from encoder import TransformerEncoderLayer

class EncoderOnlyNextTokenPredictor(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_embedding = PositionEmbedding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size) 
    
    def forward(self, x):  # x: (batch, seq_len)
        tok_emb = self.token_embedding(x)                   # (batch, seq_len, d_model)
        pos_emb = self.pos_embedding(x)                     # (1, seq_len, d_model)
        x = tok_emb + pos_emb                               # (batch, seq_len, d_model)

        for layer in self.encoder_layers:
            x = layer(x)                                     # (batch, seq_len, d_model)
        
        x = self.norm(x)                                  
        x = x.mean(dim=1)                                
        out = self.classifier(x)                             # (batch, vocab_size)

        return out 
