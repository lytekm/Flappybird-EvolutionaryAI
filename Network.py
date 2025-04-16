import torch
from torch import nn


class FlappyNet(nn.Module):
    def __init__(self, input_dim):
        super(FlappyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)