import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = "MountainCarContinuous-v0"
#параметры по умолчанию
num_iterations = 300
num_steps = 2048 #длина траектории
ppo_epochs = 10
mini_batch_size = 256
gamma = 0.99
clip_ratio = 0.2
value_coef = 0.5
entropy_coef = 0.01
lr = 3e-4


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = self.shared(x)
        return self.mean(x), self.log_std.exp()

    def get_dist(self, state):
        mean, std = self.forward(state)
        return Normal(mean, std)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        dist = self.get_dist(state)
        action = dist.sample()
        return action.cpu().numpy(), dist.log_prob(action).sum().item()


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def collect_trajectories(env, policy, steps):
    states, actions, log_probs, rewards, dones = [], [], [], [], []
    state, _ = env.reset(seed=1)
    for _ in range(steps):
        action, log_prob = policy.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)

        state = next_state
        if done:
            state, _ = env.reset()

    return map(np.array, (states, actions, log_probs, rewards, dones))


def compute_advantages(rewards, dones, values, norm):
    returns, advantages = [], []
    R = 0
    for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
        R = r + gamma * R * (1 - d)
        returns.insert(0, R)
        advantages.insert(0, R - v)
    returns, advantages = np.array(returns), np.array(advantages)
    #нормализация преимуществ
    if norm:
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


def train(num_steps, clip_ratio, ppo_epochs, norm):
    env = gym.make(env_name)
    actor = Actor(2, 1).to(device)
    critic = Critic(2).to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=lr)
    opt_critic = optim.Adam(critic.parameters(), lr=lr)

    avg_rewards = []
    total_losses = []

    for iteration in range(num_iterations):
        states, actions, log_probs, rewards, dones = collect_trajectories(env, actor, num_steps)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)
        old_log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32).to(device)
        values = critic(states_tensor).squeeze().detach().cpu().numpy()

        returns, advantages = compute_advantages(rewards, dones, values, norm)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)

        iter_losses = []
        for _ in range(ppo_epochs):
            idxs = np.arange(len(states))
            np.random.shuffle(idxs)
            for start in range(0, len(states), mini_batch_size):
                end = start + mini_batch_size
                batch_idx = idxs[start:end]

                s = states_tensor[batch_idx]
                a = actions_tensor[batch_idx]
                old_logp = old_log_probs_tensor[batch_idx]
                ret = returns[batch_idx]
                adv = advantages[batch_idx]

                dist = actor.get_dist(s)
                new_logp = dist.log_prob(a).sum(dim=-1)
                ratio = torch.exp(new_logp - old_logp)

                actor_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv).mean()
                entropy = dist.entropy().mean()
                critic_loss = (critic(s).squeeze() - ret).pow(2).mean()

                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
                iter_losses.append(loss.item())

                opt_actor.zero_grad()
                opt_critic.zero_grad()
                loss.backward()
                opt_actor.step()
                opt_critic.step()

        avg_reward = sum(rewards) / (sum(dones) if sum(dones) > 0 else 1)
        avg_rewards.append(avg_reward)
        total_losses.append(np.mean(iter_losses))

        if iteration%25 == 0:
          print(f"iteration {iteration}: avg_reward: {avg_reward:.2f}, loss: {total_losses[-1]:.4f}")

        if avg_reward >= 90:
            break


    torch.save(actor.state_dict(), f"ppo_actor_steps{num_steps}_clip{clip_ratio}_epochs{ppo_epochs}.pth")
    return avg_rewards, total_losses

def plot_results(rewards_dict, losses_dict):
    plt.figure(figsize=(10, 5))
    for name, rewards in rewards_dict.items():
        plt.plot(rewards, label=name)
    plt.title('Reward comp')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    for name, losses in losses_dict.items():
        plt.plot(losses, label=name)
    plt.title('Loss comp')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

cr_rewards = {}
cr_losses = {}

for cr in [0.1, 0.2, 0.5]:
  avg_rewards, total_losses = train(num_steps=2048, clip_ratio=cr, ppo_epochs=40, norm = False)
  cr_rewards[f"clip={cr}"] = avg_rewards
  cr_losses[f"clip={cr}"] = total_losses

plot_results(cr_rewards, cr_losses)

cr_rewards_tr = {}
cr_losses_tr = {}

for cr in [0.1, 0.2, 0.5]:
  avg_rewards, total_losses = train(num_steps=2048, clip_ratio=cr, ppo_epochs=40, norm = True)
  cr_rewards_tr[f"clip={cr}"] = avg_rewards
  cr_losses_tr[f"clip={cr}"] = total_losses

plot_results(cr_rewards_tr, cr_losses_tr)

epochs_rewards = {}
epochs_losses = {}

for ep in [10, 30, 40]:
  avg_rewards, total_losses = train(num_steps=2048, clip_ratio=0.2, ppo_epochs=ep, norm = False)
  epochs_rewards[f"epochs={ep}"] = avg_rewards
  epochs_losses[f"epochs={ep}"] = total_losses

plot_results(epochs_rewards, epochs_losses)

epochs_rewards = {}
epochs_losses = {}

for ep in [10, 30, 40]:
  avg_rewards, total_losses = train(num_steps=2048, clip_ratio=0.2, ppo_epochs=ep, norm = True)
  epochs_rewards[f"epochs={ep}"] = avg_rewards
  epochs_losses[f"epochs={ep}"] = total_losses

plot_results(epochs_rewards, epochs_losses)

step_rewards = {}
step_losses = {}

for st in [512, 1024, 2048]:
  avg_rewards, total_losses = train(num_steps=st, clip_ratio=0.2, ppo_epochs=40, norm = False)
  step_rewards[f"num_steps={st}"] = avg_rewards
  step_losses[f"num_steps={st}"] = total_losses

plot_results(step_rewards, step_losses)

step_rewards = {}
step_losses = {}

for st in [512, 1024, 2048]:
  avg_rewards, total_losses = train(num_steps=st, clip_ratio=0.2, ppo_epochs=40, norm = True)
  step_rewards[f"num_steps={st}"] = avg_rewards
  step_losses[f"num_steps={st}"] = total_losses

plot_results(step_rewards, step_losses)
