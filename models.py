import torch
from torch import nn
import torch.nn.functional as F


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    k_q = torch.bmm(query, key.permute(0, 2, 1))
    mask = torch.triu(torch.full(k_q.shape, float('-inf')), diagonal=1)
    selected = torch.softmax(k_q + mask, dim=2)

    return torch.bmm(selected, value)


class Model(nn.Module):
    def __init__(self, d_model: int, d_attention: int, vocab_size: int, padding_index: int):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, d_model, padding_idx=padding_index)
        self.decoder = Decoder(d_model, d_attention, vocab_size, 2048)

    def forward(self, x):
        x = self.embedder(x)
        x = self.decoder(x)
        return x

    def generate(self, text, tokenizer, bos=1):
        encoded = tokenizer.Encode(text)
        encoded = torch.Tensor([[bos] + encoded]).long()
        for _ in range(150):
            output = self.forward(encoded)
            output = output[:, -1, :]
            pred = output.softmax(dim=1)
            pred = pred.argmax(1).unsqueeze(dim=0)
            encoded = torch.cat([encoded, pred], dim=1)

        return tokenizer.Decode(encoded.numpy().tolist())


class Decoder(nn.Module):
    def __init__(self, d_model: int, d_attention: int, vocab_size: int, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_attention, d_ff) for _ in range(12)])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.out(x)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_attention: int, d_ff):
        super().__init__()
        self.layer1 = SelfAttention(d_model, d_attention)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = self.layer1(x)
        x = self.ff(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, d_attention: int):
        super().__init__()
        self.queries = nn.Linear(d_model, d_attention)
        self.keys = nn.Linear(d_model, d_attention)
        self.values = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        query = self.queries(x)
        key = self.keys(x)
        value = self.values(x)
        self_attention = attention(query, key, value)
        return self.norm(x + self_attention)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
