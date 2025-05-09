import random
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
from torch import nn
from torch import optim
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

os.makedirs("plots", exist_ok=True)

class ReplayBuffer:
    def __init__(self, capacity=10000):
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

class QNetworkSmall(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class QNetworkMedium(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class QNetworkLarge(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_size, n_actions, network_type='medium', gamma=0.99, epsilon_decay=0.995, epsilon_start=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if network_type == 'small':
            self.q_net = QNetworkSmall(obs_size, n_actions).to(self.device)
            self.target_net = QNetworkSmall(obs_size, n_actions).to(self.device)
        elif network_type == 'large':
            self.q_net = QNetworkLarge(obs_size, n_actions).to(self.device)
            self.target_net = QNetworkLarge(obs_size, n_actions).to(self.device)
        else:
            self.q_net = QNetworkMedium(obs_size, n_actions).to(self.device)
            self.target_net = QNetworkMedium(obs_size, n_actions).to(self.device)
            
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.gamma = gamma
        self.batch_size = 128
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.replay_buffer = ReplayBuffer(10000)
        self.network_type = network_type

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                q_values = self.q_net(state_tensor)
                return torch.argmax(q_values).item()
            
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(1)[0]
            target_q_values = reward + self.gamma * next_q_values * (1 - done)
        
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

def smooth(y, sigma=2):
    return gaussian_filter1d(y, sigma=sigma)

def plot_individual_results(results, title, xlabel, ylabel, filename, smooth_curve=True):
    plt.figure(figsize=(10, 6))
    for name, rewards in results.items():
        if smooth_curve:
            plt.plot(smooth(rewards), label=name)
        else:
            plt.plot(rewards, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/{filename}.png")
    plt.close()

def plot_separate_results(results, title_prefix, xlabel, ylabel, smooth_curve=True):
    for name, rewards in results.items():
        plt.figure(figsize=(10, 6))
        if smooth_curve:
            plt.plot(smooth(rewards))
        else:
            plt.plot(rewards)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix} ({name})")
        plt.grid()
        plt.savefig(f"plots/{title_prefix.lower().replace(' ', '_')}_{name}.png")
        plt.close()

def train_agent(params):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(**params)
    rewards = []
    
    for episode in tqdm(range(500), desc=f"Training {params.get('network_type', 'medium')}"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        for _ in range(200):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            agent.train()

            if done: break
        
        agent.update_target()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        rewards.append(episode_reward)
    
    env.close()
    return rewards

# 1. Эксперименты с архитектурами
print("Running architecture experiments...")
architectures = ['small', 'medium', 'large']
arch_results = {}

for arch in architectures:
    rewards = train_agent({
        'obs_size': 4,
        'n_actions': 2,
        'network_type': arch,
        'gamma': 0.99,
        'epsilon_decay': 0.995
    })
    arch_results[arch] = rewards

plot_individual_results(
    arch_results,
    "Comparison of Different Network Architectures",
    "Episode",
    "Reward",
    "architectures_comparison"
)

plot_separate_results(
    arch_results,
    "Network Architecture Performance",
    "Episode",
    "Reward"
)

# 2. Эксперименты с гиперпараметрами
print("\nRunning hyperparameter experiments...")

# 2.1. Разные epsilon_decay при gamma=0.99
decay_values = [0.99, 0.995, 0.999]
decay_results = {}

for decay in decay_values:
    rewards = train_agent({
        'obs_size': 4,
        'n_actions': 2,
        'gamma': 0.99,
        'epsilon_decay': decay
    })
    decay_results[f"decay={decay}"] = rewards

plot_individual_results(
    decay_results,
    "Different Epsilon Decay Values (gamma=0.99)",
    "Episode",
    "Reward",
    "epsilon_decay_comparison"
)

plot_separate_results(
    decay_results,
    "Epsilon Decay Performance",
    "Episode",
    "Reward"
)

# 2.2. Разные gamma при epsilon_decay=0.995
gamma_values = [0.95, 0.99, 0.999]
gamma_results = {}

for gamma in gamma_values:
    rewards = train_agent({
        'obs_size': 4,
        'n_actions': 2,
        'gamma': gamma,
        'epsilon_decay': 0.995
    })
    gamma_results[f"gamma={gamma}"] = rewards

plot_individual_results(
    gamma_results,
    "Different Gamma Values (epsilon_decay=0.995)",
    "Episode",
    "Reward",
    "gamma_comparison"
)

plot_separate_results(
    gamma_results,
    "Gamma Value Performance",
    "Episode",
    "Reward"
)

# 3. Эксперименты с начальным epsilon (5 значений)
print("\nRunning epsilon start experiments...")
epsilon_starts = [0.2, 0.5, 1.0, 1.5, 2.0]
epsilon_results = {}

for eps in epsilon_starts:
    rewards = train_agent({
        'obs_size': 4,
        'n_actions': 2,
        'gamma': 0.99,
        'epsilon_decay': 0.995,
        'epsilon_start': eps
    })
    epsilon_results[f"eps_start={eps}"] = rewards

plot_individual_results(
    epsilon_results,
    "Different Initial Epsilon Values",
    "Episode",
    "Reward",
    "epsilon_start_comparison"
)

plot_separate_results(
    epsilon_results,
    "Initial Epsilon Performance",
    "Episode",
    "Reward"
)

print("\nAll experiments completed! Results saved to 'plots' folder.")