"""ANN and DNN architectures (unchanged from v1 train_final.py)."""
import torch.nn as nn


class FloodANN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
                                   nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.model(x)


class FloodDNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.4),
                                   nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
                                   nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.model(x)
