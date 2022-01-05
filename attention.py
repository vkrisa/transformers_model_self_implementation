import torch
from torch import nn

SEQUENCE_LENGTH = 3
D_MODEL = 64
D_ATTENTION = 32


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    k_q = torch.matmul(query, key.T)

    mask_positions = torch.triu_indices(*k_q.shape, offset=1)
    attention_mask = torch.zeros_like(k_q)
    attention_mask[mask_positions[0], mask_positions[1]] = float('-inf')

    selected = torch.softmax(k_q + attention_mask, dim=1)
    return torch.matmul(selected, value)


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, d_attention: int):
        super(SelfAttention, self).__init__()
        self.queries = nn.Linear(d_model, d_attention)
        self.keys = nn.Linear(d_model, d_attention)
        self.values = nn.Linear(d_model, d_model)

    def forward(self, x):
        query = self.queries(x)
        key = self.keys(x)
        value = self.values(x)
        self_attention = attention(query, key, value)
        return x + self_attention


if __name__ == '__main__':
    X = torch.randn(SEQUENCE_LENGTH, D_MODEL)
    model = SelfAttention(D_MODEL, D_ATTENTION)
    output = model(X)
    print(output)