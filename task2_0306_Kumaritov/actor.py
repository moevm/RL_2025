import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_size=256):
        super(Actor, self).__init__()
        self.log_std = nn.Parameter(torch.zeros(n_actions))
        self.model = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, n_actions)
        )

    def forward(self, x):
        return self.model(x)

    def get_dist(self, x):
        mean = self.forward(x)
        std = torch.exp(self.log_std).clamp(1e-6, 1)
        return Normal(mean, std)

    def get_action(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        dist_x = self.get_dist(x)
        action = dist_x.rsample()
        log_prob = dist_x.log_prob(action).sum(-1)
        return action, log_prob
