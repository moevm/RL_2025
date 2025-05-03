import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import trange

# --- Настройки и архитектуры ---
params = {
    "gamma": 0.99,
    "epsilon": 0.9,
    "epsilon_decay": 0.955,
    "epsilon_min": 0.05,
    "batch_size": 128,
    "num_steps": 200,
    "num_episodes": 300,
    "lr": 1e-4,
}

architectures = {
    "default": [128, 64],
    "large": [256, 128, 64],
    "small": [32],
    "deep": [64, 64, 64, 32]
}


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layers):
        super().__init__()
        net = []
        last_dim = input_dim
        for l in layers:
            net.append(nn.Linear(last_dim, l))
            net.append(nn.ReLU())
            last_dim = l
        net.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, layer_cfg, gamma, epsilon, epsilon_decay):
        self.q_net = QNetwork(state_dim, action_dim, layer_cfg)
        self.target_net = QNetwork(state_dim, action_dim, layer_cfg)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=params["lr"])
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.q_net(state_tensor).argmax().item()

    def train_step(self):
        if len(self.buffer) < params["batch_size"]:
            return 0
        s, a, r, s2, d = self.buffer.sample(params["batch_size"])
        s, a, r, s2, d = s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device)

        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * self.target_net(s2).max(1)[0] * (1 - d)
        loss = nn.MSELoss()(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, params["epsilon_min"])

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


def train(agent, env, episodes, steps):
    reward_history = []
    loss_history = []
    for ep in trange(episodes, desc="Эпизоды"):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        for _ in range(steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            state = next_state
            total_reward += reward
            total_loss += loss
            if done or truncated:
                break
        reward_history.append(total_reward)
        loss_history.append(total_loss)
        agent.update_epsilon()
        agent.update_target()
    return reward_history, loss_history


def plot_results(results, title, filename):
    plt.figure(figsize=(10, 4))
    for label, (rewards, _) in results.items():
        plt.plot(rewards, label=label)
    plt.title(f"{title} — Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{filename}_reward.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    for label, (_, losses) in results.items():
        plt.plot(losses, label=label)
    plt.title(f"{title} — Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"figs/{filename}_loss.png")
    plt.close()


def experiment_default():
    env = gym.make("CartPole-v1")
    agent = DQNAgent(4, 2, architectures["default"], params["gamma"], params["epsilon"], params["epsilon_decay"])
    rewards, losses = train(agent, env, params["num_episodes"], params["num_steps"])
    return {"default": (rewards, losses)}


def experiment_architectures():
    env = gym.make("CartPole-v1")
    results = {}
    for name, layers in architectures.items():
        agent = DQNAgent(4, 2, layers, params["gamma"], params["epsilon"], params["epsilon_decay"])
        rewards, losses = train(agent, env, params["num_episodes"], params["num_steps"])
        results[name] = (rewards, losses)
    return results


def experiment_gamma_decay():
    env = gym.make("CartPole-v1")
    results = {}
    for gamma in [0.99, 0.9, 0.8, 0.7]:
        for decay in [0.995, 0.95, 0.9, 0.8]:
            label = f"g={gamma}, d={decay}"
            agent = DQNAgent(4, 2, architectures["default"], gamma, params["epsilon"], decay)
            rewards, losses = train(agent, env, params["num_episodes"], params["num_steps"])
            results[label] = (rewards, losses)
    return results


def experiment_epsilons():
    env = gym.make("CartPole-v1")
    results = {}
    for eps in [0.9, 0.7, 0.5, 0.3, 0.1]:
        agent = DQNAgent(4, 2, architectures["default"], params["gamma"], eps, params["epsilon_decay"])
        rewards, losses = train(agent, env, params["num_episodes"], params["num_steps"])
        results[f"eps={eps}"] = (rewards, losses)
    return results

def experiment_gamma_decay_grouped():
    env = gym.make("CartPole-v1")
    grouped_results = {}
    for gamma in [0.99, 0.9, 0.8, 0.7]:
        sub_results = {}
        for decay in [0.995, 0.95, 0.9, 0.8]:
            label = f"decay={decay}"
            agent = DQNAgent(4, 2, architectures["default"], gamma, params["epsilon"], decay)
            rewards, losses = train(agent, env, params["num_episodes"], params["num_steps"])
            sub_results[label] = (rewards, losses)
        grouped_results[f"gamma={gamma}"] = sub_results
    return grouped_results

def plot_results_grouped(grouped_results, base_filename):
    os.makedirs("figs", exist_ok=True)
    for group_name, results in grouped_results.items():
        # Reward
        plt.figure(figsize=(10, 4))
        for label, (rewards, _) in results.items():
            plt.plot(rewards, label=label)
        plt.title(f"{group_name} — Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid()
        plt.savefig(f"figs/{base_filename}_{group_name}_reward.png")
        plt.close()

        # Loss
        plt.figure(figsize=(10, 4))
        for label, (_, losses) in results.items():
            plt.plot(losses, label=label)
        plt.title(f"{group_name} — Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"figs/{base_filename}_{group_name}_loss.png")
        plt.close()


if __name__ == "__main__":
    print("Running default experiment...")
    r_default = experiment_default()
    plot_results(r_default, "Default", "default")

    print("Running architecture experiment...")
    r_arch = experiment_architectures()
    plot_results(r_arch, "Architectures", "architectures")

    print("Running gamma & decay experiment...")
    r_gd = experiment_gamma_decay()
    plot_results(r_gd, "Gamma and Epsilon Decay", "gamma_decay")

    print("Running epsilon init experiment...")
    r_eps = experiment_epsilons()
    plot_results(r_eps, "Initial Epsilon", "epsilon_init")

    print("Running improved gamma & decay experiment...")
    r_grouped = experiment_gamma_decay_grouped()
    plot_results_grouped(r_grouped, "gamma_decay_grouped")



