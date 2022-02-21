import random

import torch
import numpy as np
from models import Model
from tqdm import tqdm
import sentencepiece as spm
from dataset import LanguageDataset
from torch.utils.data import DataLoader

INPUT_PATH = "/home/vargak/work/datasets/lm/"
D_MODEL = 512
D_ATTENTION = 512
N_BPE = 4096
BATCH_SIZE = 16
EPOCH_SIZE = 10
PADDING_INDEX = 0
SEQUENCE_LENGTH = 256
BOS = 1
device = torch.device('cuda:0')


def padding_to_longest(data: [[int]]) -> torch.Tensor:
    longest = max(len(row) for row in data)
    padded = np.zeros((len(data), longest))
    for idx, row in enumerate(data):
        padded[idx, :len(row)] += row

    return torch.Tensor(padded).long().to(device)


def slice_sequence(row: [int]) -> [int]:
    if len(row) > SEQUENCE_LENGTH:
        index = random.randint(0, len(row) - SEQUENCE_LENGTH)
        row = row[index:index + SEQUENCE_LENGTH]
    return row


def save_model(model, path):
    torch.save(model.state_dict(), path)


def collate(batch):
    encoded = [tokenizer.Encode(row) for row in batch]
    encoded = [[BOS] + row for row in encoded]
    sliced = [slice_sequence(row) for row in encoded]
    padded = padding_to_longest(sliced)
    item = padded[:, :-1]
    label = padded[:, 1:]
    return item, label


if __name__ == '__main__':
    tokenizer = spm.SentencePieceProcessor('tokenizer/hu.model')
    model = Model(D_MODEL, D_ATTENTION, N_BPE, PADDING_INDEX).cuda()
    train_set = LanguageDataset(INPUT_PATH)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate)

    learning_rate = 2e-4
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PADDING_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(EPOCH_SIZE):
        losses = []
        for batch in tqdm(train_loader):
            item, label = batch

            # train step
            model.train()
            y_prediction = model(item)
            loss = loss_fn(y_prediction.flatten(0, 1), label.flatten(0, 1))
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'loss: {sum(losses) / len(losses)}')
        save_model(model, f'output/{epoch}')
