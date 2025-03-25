import torch
import torch.nn as nn
from transformers import AutoModel

class SimCSEModel(nn.Module):
    def __init__(self, model_name: str):
        super(SimCSEModel, self).__init__()
        # load BERT
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dense = nn.Linear(self.backbone.config.hidden_size, self.backbone.config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state[:, 0]  # [CLS] 토큰 추출
        emb = self.dense(emb)
        emb = self.activation(emb)
        return emb
