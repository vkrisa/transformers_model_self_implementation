import torch
from torch import nn


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    k_q = torch.bmm(query, key.permute(0, 2, 1))
    mask = torch.triu(torch.full(k_q.shape, float('-inf')), diagonal=1)
    selected = torch.softmax(k_q + mask, dim=2)

    return torch.bmm(selected, value)


class Model(nn.Module):
    def __init__(self, d_model: int, d_attention: int, vocab_size: int, padding_index: int):
        super().__init__()
        self.d_model = d_model

        self.embedder = nn.Embedding(vocab_size, d_model, padding_idx=padding_index)
        self.layer1 = SelfAttention(d_model, d_attention)
        self.layer2 = SelfAttention(d_model, d_attention)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedder(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.out(x)
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
