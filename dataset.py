import torch
import pathlib
import os


class LanguageDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        directory = pathlib.Path(root)
        self.data = [self.read(f'{str(f)}/texts.txt') for f in directory.iterdir() if os.path.exists(f'{str(f)}/texts.txt')]
        self.data = [item for sublist in self.data for item in sublist]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def read(self, path):
        with open(path, "r") as f:
            return f.readlines()
