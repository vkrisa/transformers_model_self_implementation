import torch
import numpy as np
from models import Model

D_MODEL = 64
D_ATTENTION = 32
N_BPE = 8
PADDING_INDEX = 0


def train_step():
    model.train()
    y_prediction = model(X[:, :-1])
    loss = loss_fn(y_prediction.flatten(0, 1), X[:, 1:].flatten(0, 1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_step():
    model.eval()
    output = model(X[:, :-1])
    prediction = torch.softmax(output, dim=2)
    prediction = torch.argmax(prediction, 2)
    if epoch % 500 == 499:
        print(f"epoch: {epoch}")
        print(f"pred: {prediction}")


def prepare_data(data: [[int]]):
    longest = max(len(row) for row in data)
    padded = np.zeros((len(data), longest))

    for idx, row in enumerate(data):
        padded[idx, :len(row)] += row

    return torch.Tensor(padded).long()


if __name__ == '__main__':
    train_data = [
        [1, 2, 3, 4, 5, 6, 7],
        [1, 6, 5, 4, 3, 7],
        [1, 2, 3, 7]
    ]
    X = prepare_data(train_data)
    model = Model(D_MODEL, D_ATTENTION, N_BPE, PADDING_INDEX)

    learning_rate = 2e-4
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PADDING_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch = 0
    while True:
        train_step()
        test_step()
        epoch += 1
