import gymnasium as gym
import torch
import numpy as np
from collections import deque
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import os

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions, arch_type="default"):
        super(QNetwork, self).__init__()
        if arch_type == "shallow":
            self.net = nn.Sequential(
                nn.Linear(obs_size, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions)
            )
        elif arch_type == "default":
            self.net = nn.Sequential(
                nn.Linear(obs_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions)
            )
        elif arch_type == "wide":
            self.net = nn.Sequential(
                nn.Linear(obs_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, n_actions)
            )
        elif arch_type == "deep":
            self.net = nn.Sequential(
                nn.Linear(obs_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions)
            )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_size, n_actions, arch_type="default", gamma=0.99, epsilon=1.0, epsilon_decay=0.955):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(obs_size, n_actions, arch_type).to(self.device)
        self.target_net = QNetwork(obs_size, n_actions, arch_type).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)

        self.gamma = gamma
        self.batch_size = 64
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01

        self.replay_buffer = ReplayBuffer(1000)

        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                q_values = self.q_net(state_tensor)
                return torch.argmax(q_values).item()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        target_q_values = reward + self.gamma * self.target_net(next_state).max(1)[0] * (1 - done)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

def run_experiment(arch_type="default", gamma=0.99, epsilon=1.0, epsilon_decay=0.955, episodes=300, label=None):
    env = gym.make("CartPole-v1", render_mode=None)
    agent = DQNAgent(
        obs_size=4,
        n_actions=2,
        arch_type=arch_type,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay
    )

    rewards = []
    losses = []

    for episode in tqdm(range(episodes), desc=label):
        state, _ = env.reset()
        total_reward = 0
        episode_losses = []

        for _ in range(200):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.update_target()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        rewards.append(total_reward)
        losses.append(np.mean(episode_losses) if episode_losses else 0)

    env.close()
    return rewards, losses


def plot_experiment_results(param_name, values, param_key):
    results = {}

    for val in values:
        kwargs = {param_key: val}
        rewards, losses = run_experiment(**kwargs, label=f"{param_name}: {val}")
        results[val] = {"rewards": rewards, "losses": losses}

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12, 6))
    for val in values:
        plt.plot(results[val]["rewards"], label=f"{param_name}={val}")
    plt.title(f"Влияние {param_name} на обучение DQN (Reward)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    filename_reward = f"plots/reward_vs_{param_name}.png"
    plt.savefig(filename_reward)
    # plt.show()

    plt.figure(figsize=(12, 6))
    for val in values:
        plt.plot(results[val]["losses"], label=f"{param_name}={val}")
    plt.title(f"Влияние {param_name} на обучение DQN (Loss)")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    filename_loss = f"plots/loss_vs_{param_name}.png"
    plt.savefig(filename_loss)
    # plt.show()

plot_experiment_results("default", ["default"], "arch_type")

plot_experiment_results("architecture", ["shallow", "default", "deep", "wide"], "arch_type")

plot_experiment_results("gamma", [0.90, 0.95, 0.98, 0.99], "gamma")

plot_experiment_results("epsilon", [1.0, 0.5, 0.2], "epsilon")

plot_experiment_results("epsilon_decay", [0.99, 0.97, 0.95, 0.90], "epsilon_decay")
