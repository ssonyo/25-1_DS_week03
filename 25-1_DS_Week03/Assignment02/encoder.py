import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple



class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1) 
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) 
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        result = torch.matmul(attention, v)
        return result, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = Q.size(0)
        
        Q = self.query_layers(Q).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        K = self.key_layers(K).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        V = self.value_layers(V).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        
        x, attention = self.attention(Q, K, V, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_model)
        
        x = self.fc(x)
        return x
    

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(F.relu(self.linear1(x)))

class DropoutLayer(nn.Module):
    def __init__(self, p: float) -> None:
        super(DropoutLayer, self).__init__()
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)

class ActivationLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(LayerNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.layer_norm(x)
    
class ResidualConnection(nn.Module):
    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:

        return x + sublayer
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForwardLayer(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
    
    def forward(self, x: Tensor) -> Tensor:
        mask = None

        self_attention = self.self_attn(x, x, x, mask)
        x = self.residual1(x, self.dropout1(self_attention))
        x = self.norm1(x)
        
        ffn = self.ff(x)
        x = self.residual2(x, self.dropout2(ffn))
        x = self.norm2(x)
        
        return x