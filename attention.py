import torch
from models import Model

SEQUENCE_LENGTH = 8
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
    if epoch % 500 == 0:
        print(f"epoch: {epoch}")
        print(f"target: {d}")
        print(f"pred: {prediction}")


if __name__ == '__main__':
    d = [[1, 2, 3, 4, 5, 6, 7],
         [1, 6, 5, 4, 3, 2, 7]]
    X = torch.Tensor(d).long()
    model = Model(D_MODEL, D_ATTENTION, N_BPE)

    learning_rate = 2e-4
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch = 0
    while True:
        train_step()
        test_step()
        epoch += 1
