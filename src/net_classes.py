import torch
import torch.nn as nn

class LinearNet(nn.Module):
    def __init__(self, size, hidden=2048):
        super(LinearNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(size, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, 200),
            nn.Softmax()
        )

    def forward(self, x):
        return self.fc(x)

