import torch


class LanguageDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = self.read(path)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def read(self, path):
        with open(path, "r") as f:
            return f.readlines()