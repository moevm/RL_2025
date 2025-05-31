import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_size):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)


class Critic_DoubleQ(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_size):
        super(Critic_DoubleQ, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

        self.q2 = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.q1(x), self.q2(x)
