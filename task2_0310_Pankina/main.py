import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plot_dir = "plots_pro"
os.makedirs(plot_dir, exist_ok=True)

env_name = "MountainCarContinuous-v0"
num_iterations = 999
num_steps = 2048
ppo_epochs = 10
mini_batch_size = 64
gamma = 0.99
clip_ratio = 0.2
value_coef = 0.5
entropy_coef = 0.01
lr = 3e-4


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        features = self.net(x)
        mean = self.mean(features)
        return mean, self.log_std.exp()

    def get_dist(self, state):
        mean, std = self.forward(state)
        return Normal(mean, std)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            dist = self.get_dist(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy().flatten(), log_prob.item()


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.net(state)


def collect_trajectories(policy, num_steps):
    env = gym.make(env_name)
    states, actions, log_probs, rewards, dones, episode_rewards = [], [], [], [], [], []
    state, _ = env.reset()
    ep_reward = 0.0

    for _ in range(num_steps):
        action, log_prob = policy.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)

        state = next_state
        ep_reward += float(reward)

        if done:
            state, _ = env.reset()
            episode_rewards.append(ep_reward)
            ep_reward = 0.0

    if len(episode_rewards) == 0 or ep_reward > 0:
        episode_rewards.append(ep_reward)

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "log_probs": np.array(log_probs),
        "rewards": np.array(rewards),
        "dones": np.array(dones),
        "episode_rewards": np.array(episode_rewards),
    }


def compute_returns_advantages(rewards, dones, values, normalize_advantages=True):
    returns = []
    advantages = []
    R = 0.0

    for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
        if done:
            R = 0.0
        R = reward + gamma * R
        returns.insert(0, R)
        advantages.insert(0, R - value)

    returns = np.array(returns)
    advantages = np.array(advantages)

    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


def train(env, actor, critic, num_iterations, num_steps, ppo_epochs, clip_ratio, normalize_advantages):
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    all_avg_rewards = []

    for i in tqdm(range(num_iterations)):
        batch = collect_trajectories(actor, num_steps)
        states = torch.FloatTensor(batch["states"]).to(device)
        actions = torch.FloatTensor(batch["actions"]).to(device)
        old_log_probs = torch.FloatTensor(batch["log_probs"]).to(device)

        with torch.no_grad():
            values = critic(states).squeeze().cpu().numpy()

        returns, advantages = compute_returns_advantages(batch["rewards"], batch["dones"], values, normalize_advantages)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        for epoch in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, mini_batch_size):
                end = min(start + mini_batch_size, dataset_size)
                mini_indices = indices[start:end]

                mini_states = states[mini_indices]
                mini_actions = actions[mini_indices]
                mini_old_log_probs = old_log_probs[mini_indices]
                mini_returns = returns[mini_indices]
                mini_advantages = advantages[mini_indices]

                dist = actor.get_dist(mini_states)
                new_log_probs = dist.log_prob(mini_actions).sum(dim=-1)
                ratio = torch.exp(new_log_probs - mini_old_log_probs)
                surrogate1 = ratio * mini_advantages
                surrogate2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mini_advantages

                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                entropy_loss = dist.entropy().mean()
                value_estimates = critic(mini_states).squeeze()
                critic_loss = (mini_returns - value_estimates).pow(2).mean()

                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()

        avg_reward = np.mean(batch["episode_rewards"])
        # print(f"Iteration {i + 1}: avg_reward = {avg_reward:.2f}")
        all_avg_rewards.append(avg_reward)

        if avg_reward >= 90:
            print("Задача выполнена!")
            break

    return all_avg_rewards


def run_experiment(param_name, param_values, param_label, **train_kwargs):
    all_results = {}

    for value in param_values:
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        actor = Actor(state_dim, action_dim).to(device)
        critic = Critic(state_dim).to(device)

        kwargs = train_kwargs.copy()
        kwargs[param_name] = value
        rewards = train(env, actor, critic, **kwargs)
        all_results[str(value)] = rewards

    plt.figure(figsize=(10, 5))
    for label, rewards in all_results.items():
        plt.plot(rewards, label=f"{param_label}={label}")
    plt.xlabel("Итерации")
    plt.ylabel("Средняя награда")
    plt.title(f"Сравнение по параметру: {param_label}")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    filename = f"{param_label}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath)
    plt.show()


run_experiment("num_steps", [1024, 2048, 4096], "num_steps",
               num_iterations=num_iterations,
               ppo_epochs=ppo_epochs,
               clip_ratio=clip_ratio,
               normalize_advantages=False)

run_experiment("clip_ratio", [0.1, 0.2, 0.3], "clip_ratio",
               num_iterations=num_iterations,
               num_steps=num_steps,
               ppo_epochs=ppo_epochs,
               normalize_advantages=False)

run_experiment("normalize_advantages", [True, False], "normalize_advantages",
               num_iterations=num_iterations,
               num_steps=num_steps,
               ppo_epochs=ppo_epochs,
               clip_ratio=clip_ratio)

run_experiment("ppo_epochs", [5, 10, 20], "ppo_epochs",
               num_iterations=num_iterations,
               num_steps=num_steps,
               clip_ratio=clip_ratio,
               normalize_advantages=False)
