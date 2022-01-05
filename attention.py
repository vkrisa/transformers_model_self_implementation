import torch
from models import Model

SEQUENCE_LENGTH = 8
D_MODEL = 64
D_ATTENTION = 32
N_BPE = 8


def train_step():
    model.train()
    y_prediction = model(X)
    loss = loss_fn(y_prediction, X)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_step():
    model.eval()
    output = model(X[:, :-1])
    prediction = torch.softmax(output, dim=1)
    prediction = torch.argmax(prediction, 1)
    print(prediction)


if __name__ == '__main__':
    X = torch.Tensor([[0, 1, 2, 3, 4, 5, 6, 7]]).long()
    model = Model(D_MODEL, D_ATTENTION, N_BPE)

    learning_rate = 1e-3
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    while True:
        # train_step()
        test_step()
