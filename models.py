import torch
from torch import nn


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    k_q = torch.matmul(query, key.T)

    mask_positions = torch.triu_indices(*k_q.shape, offset=1)
    attention_mask = torch.zeros_like(k_q)
    attention_mask[mask_positions[0], mask_positions[1]] = float('-inf')

    selected = torch.softmax(k_q + attention_mask, dim=1)
    return torch.matmul(selected, value)


class Model(nn.Module):
    def __init__(self, d_model: int, d_attention: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model

        self.layer1 = SelfAttention(d_model, d_attention)
        self.layer2 = SelfAttention(d_model, d_attention)

        self.embedder = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = self.embedder(x)
        x = x.squeeze(0)
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, d_attention: int):
        super().__init__()
        self.queries = nn.Linear(d_model, d_attention)
        self.keys = nn.Linear(d_model, d_attention)
        self.values = nn.Linear(d_model, d_model)

    def forward(self, x):
        query = self.queries(x)
        key = self.keys(x)
        value = self.values(x)
        self_attention = attention(query, key, value)
        return x + self_attention
