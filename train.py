import torch
import numpy as np
from models import Model
from tqdm import tqdm
import sentencepiece as spm
from dataset import LanguageDataset
from torch.utils.data import DataLoader

INPUT_PATH = "data.txt"
D_MODEL = 64
D_ATTENTION = 32
N_BPE = 4096
BATCH_SIZE = 20
EPOCH_SIZE = 500
PADDING_INDEX = 0
BOS = 1
EOS = 2


def padding_to_longest(data: [[int]]) -> torch.Tensor:
    longest = max(len(row) for row in data)
    padded = np.zeros((len(data), longest))

    for idx, row in enumerate(data):
        padded[idx, :len(row)] += row

    return torch.Tensor(padded).long()


def collate(batch):
    encoded = [tokenizer.Encode(row) for row in batch]
    encoded = [[BOS] + row + [EOS] for row in encoded]
    data = padding_to_longest(encoded)
    item = data[:, :-1]
    label = data[:, 1:]
    return item, label


if __name__ == '__main__':
    tokenizer = spm.SentencePieceProcessor('tokenizer/hu.model')
    model = Model(D_MODEL, D_ATTENTION, N_BPE, PADDING_INDEX)

    train_set = LanguageDataset(INPUT_PATH)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate)

    learning_rate = 2e-4
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PADDING_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(EPOCH_SIZE)):
        for batch in train_loader:
            item, label = batch

            # train step
            model.train()
            y_prediction = model(item)
            loss = loss_fn(y_prediction.flatten(0, 1), label.flatten(0, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    result = model.generate("Most már", tokenizer)
    print(result)


