import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, n_observations, hidden_size=128):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        return self.model(x)
