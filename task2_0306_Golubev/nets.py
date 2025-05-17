import torch
import torch.nn as nn
from torch.distributions import Normal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenSize=64):
        super(Actor, self).__init__()

        self.log_std = nn.Parameter(torch.zeros(actionDim))
        self.net = nn.Sequential(
            nn.Linear(stateDim, hiddenSize),
            nn.Tanh(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.Tanh(),
            nn.Linear(hiddenSize, actionDim)
        )

    def forward(self, x):
        logits = self.net(x)

        return logits


    def get_dist(self, state):
        logits = self.forward(state)

        return Normal(loc=logits, scale=torch.exp(self.log_std))


    def commit_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist = self.get_dist(state)
        action = dist.sample()

        return  action.item(), dist.log_prob(action).item()


class Critic(nn.Module):
    def __init__(self, stateDim, hiddenSize=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(stateDim, hiddenSize),
            nn.Tanh(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.Tanh(),
            nn.Linear(hiddenSize, 1)
        )
    

    def forward(self, state):
        return self.net(state)