import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import os
import time
from itertools import product

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Actor(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenSize=64):
        super(Actor, self).__init__()
        self.log_std = nn.Parameter(torch.zeros(actionDim))
        self.mu = nn.Linear(hiddenSize, actionDim)
        self.net = nn.Sequential(
            nn.Linear(stateDim, hiddenSize),
            nn.Tanh(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.net(x)
        return self.mu(x)

    def get_dist(self, state):
        mu = self.forward(state)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def act(self, state):
        state = torch.FloatTensor(state).to(device)
        dist = self.get_dist(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.detach().cpu().numpy(), log_prob.detach().item()

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
        return self.net(state).squeeze(-1)

def collect_trajectories(policy, env, numSteps):
    states, actions, rewards, dones, logProbs = [], [], [], [], []
    episodeRewards = []
    state, _ = env.reset()
    episodeReward = 0
    for _ in range(numSteps):
        action, logProb = policy.act(state)
        nextState, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        logProbs.append(logProb)
        episodeReward += reward
        state = nextState

        if done:
            state, _ = env.reset()
            episodeRewards.append(episodeReward)
            episodeReward = 0

    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'logProbs': np.array(logProbs),
        'episodeRewards': np.array(episodeRewards)
    }

def calculate_ra(rewards, dones, values, gamma, gaeLambda):
    returns = []
    advantages = []
    R = 0
    A = 0
    next_value = 0
    for i in reversed(range(len(rewards))):
        mask = 1.0 - dones[i]
        delta = rewards[i] + gamma * next_value * mask - values[i]
        A = delta + gamma * gaeLambda * A * mask
        R = rewards[i] + gamma * R * mask
        returns.insert(0, R)
        advantages.insert(0, A)
        next_value = values[i]

    returns = np.array(returns)
    advantages = np.array(advantages)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages

numIterations = 300
numSteps = 2048
ppoEpochs = 10
miniBatchSize = 64
gamma = 0.99
gaeLambda = 0.95
clipRatio = 0.2
valueCoef = 0.5
entropyCoef = 0.01
lr = 3e-4

env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)

stateDim = env.observation_space.shape[0]
actionDim = env.action_space.shape[0]

def run_ppo_experiment(numSteps, clipRatio, normalize_advantages, ppoEpochs):
    actor = Actor(stateDim, actionDim).to(device)
    critic = Critic(stateDim).to(device)
    actorOptimizer = optim.Adam(actor.parameters(), lr=lr)
    criticOptimizer = optim.Adam(critic.parameters(), lr=lr)
    avg_rewards_history = []
    for iteration in range(numIterations):
        iterationData = collect_trajectories(actor, env, numSteps)
        states = torch.FloatTensor(iterationData['states']).to(device)
        actions = torch.FloatTensor(iterationData['actions']).to(device)
        oldLogProbs = torch.FloatTensor(iterationData['logProbs']).to(device)

        with torch.no_grad():
            values = critic(states).cpu().numpy()

        returns, advantages = calculate_ra(iterationData['rewards'], iterationData['dones'], values, gamma, gaeLambda)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        if not normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        datasetSize = states.size(0)
        indices = np.arange(datasetSize)
        for epoch in range(ppoEpochs):
            np.random.shuffle(indices)
            for start in range(0, datasetSize, miniBatchSize):
                end = start + miniBatchSize
                miniIndices = indices[start:end]
                miniStates = states[miniIndices]
                miniActions = actions[miniIndices]
                miniOldLogProbs = oldLogProbs[miniIndices]
                miniReturns = returns[miniIndices]
                miniAdvantages = advantages[miniIndices]
                dist = actor.get_dist(miniStates)
                newLogProbs = dist.log_prob(miniActions).sum(axis=-1)
                ratio = torch.exp(newLogProbs - miniOldLogProbs)
                surrogate1 = ratio * miniAdvantages
                surrogate2 = torch.clamp(ratio, 1 - clipRatio, 1 + clipRatio) * miniAdvantages
                actorLoss = -torch.min(surrogate1, surrogate2).mean()
                entropyLoss = dist.entropy().mean()
                valueEstimates = critic(miniStates)
                criticLoss = (miniReturns - valueEstimates).pow(2).mean()
                loss = actorLoss + valueCoef * criticLoss - entropyCoef * entropyLoss
                actorOptimizer.zero_grad()
                criticOptimizer.zero_grad()
                loss.backward()
                actorOptimizer.step()
                criticOptimizer.step()

        avgReward = np.mean(iterationData['episodeRewards']) if len(iterationData['episodeRewards']) > 0 else 0
        avg_rewards_history.append(avgReward)
        print(f"Iter: {iteration}, Loss: {loss.item():.4f}, AvgReward: {avgReward:.2f}")
        if avgReward >= 90:
            print("Task is considered solved.")
            break

    env.close()
    return avg_rewards_history

def run_all_experiments():
    os.makedirs("results", exist_ok=True)

    default_steps = 2048
    default_clip = 0.2
    default_epochs = 10

    def plot_summary(results_dict, param_name):
        plt.figure()
        for label, rewards in results_dict.items():
            plt.plot(rewards, label=f"{param_name}={label}")

        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.title(f"Comparison of {param_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results/summary_{param_name}.png")
        plt.close()
        print(f"Saved summary plot to results/summary_{param_name}.png")

    numSteps_list = [1024, 2048, 4096]
    step_results = {}
    for steps in numSteps_list:
        print(f"\n=== Experiment: steps={steps}, clipRatio={default_clip}, epochs={default_epochs} ===")
        rewards = run_ppo_experiment(numSteps=steps,
            clipRatio=default_clip,
            normalize_advantages=True,
            ppoEpochs=default_epochs)
        step_results[steps] = rewards
        plt.figure()
        plt.plot(rewards)
        plt.title(f"Rewards\nsteps={steps}")
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        filename = f"results/rewards_steps{steps}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot to {filename}")

    plot_summary(step_results, "steps")

    clipRatios = [0.1, 0.2, 0.3]
    clip_results = {}
    for clip in clipRatios:
        print(f"\n=== Experiment: steps={default_steps}, clipRatio={clip}, epochs={default_epochs} ===")
        rewards = run_ppo_experiment(numSteps=default_steps,
            clipRatio=clip,
            normalize_advantages=True,
            ppoEpochs=default_epochs)
        clip_results[clip] = rewards
        plt.figure()
        plt.plot(rewards)
        plt.title(f"Rewards\nclipRatio={clip}")
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        filename = f"results/rewards_clip{clip}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot to {filename}")

    plot_summary(clip_results, "clipRatio")

    ppoEpochs_list = [3, 5, 10]
    epoch_results = {}
    for epochs in ppoEpochs_list:
        print(f"\n=== Experiment: steps={default_steps}, clipRatio={default_clip}, epochs={epochs} ===")
        rewards = run_ppo_experiment(numSteps=default_steps,
            clipRatio=default_clip,
            normalize_advantages=True,
            ppoEpochs=epochs)
        epoch_results[epochs] = rewards
        plt.figure()
        plt.plot(rewards)
        plt.title(f"Rewards\nepochs={epochs}")
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        filename = f"results/rewards_epochs{epochs}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot to {filename}")

    plot_summary(epoch_results, "epochs")

if __name__ == "__main__":
    run_all_experiments()
