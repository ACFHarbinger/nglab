import logging

import torch
from torch import nn
from torch.utils.data import Dataset

from python.src.pipeline.distributed_train import train_ddp

logging.basicConfig(level=logging.INFO)


class DummyDataset(Dataset):
    def __init__(self, size=100, dim=10):
        self.data = torch.randn(size, dim)
        self.labels = torch.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleModel(nn.Module):
    def __init__(self, dim=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


def verify_ddp():
    dataset = DummyDataset()
    model = SimpleModel()
    criterion = nn.MSELoss()

    def optimizer_factory(m):
        return torch.optim.Adam(m.parameters(), lr=1e-3)

    train_ddp(
        model=model,
        dataset=dataset,
        optimizer_factory=optimizer_factory,
        criterion=criterion,
        batch_size=16,
        n_epochs=5,
    )


if __name__ == "__main__":
    verify_ddp()
