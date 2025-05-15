import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from torch.distributions import Normal
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_size, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, state):
        features = self.shared_layers(state)
        return torch.tanh(self.mu_layer(features))
    
    def get_distribution(self, state):
        mu = self.forward(state)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return Normal(mu, std)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state):
        return self.net(state).squeeze()

class PPOTrainer:
    def __init__(self, env_name, config):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.config = config
        
        self.policy = PolicyNetwork(self.state_dim, self.action_dim).to(DEVICE)
        self.value_net = ValueNetwork(self.state_dim).to(DEVICE)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config['lr'])
        self.value_optim = optim.Adam(self.value_net.parameters(), lr=config['lr'])
        
        self.rewards_history = []
        
    def collect_trajectories(self):
        states, actions, rewards, dones, log_probs = [], [], [], [], []
        state, _ = self.env.reset()
        ep_rewards = []
        current_ep_reward = 0
        
        for _ in range(self.config['num_steps']):
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            with torch.no_grad():
                dist = self.policy.get_distribution(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                
            next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = terminated or truncated
            
            states.append(state)
            actions.append(action.cpu().numpy())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            current_ep_reward += reward
            state = next_state
            
            if done:
                ep_rewards.append(current_ep_reward)
                state, _ = self.env.reset()
                current_ep_reward = 0
                
        return states, actions, rewards, dones, log_probs, ep_rewards
    
    def compute_advantages(self, rewards, dones, values, normalize=True):
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        R = 0
        
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.config['gamma'] * R * (1 - dones[t])
            returns[t] = R
            advantages[t] = R - values[t]
            
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return returns, advantages
    
    def update_networks(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.FloatTensor(np.array(actions)).to(DEVICE)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(DEVICE)
        returns = torch.FloatTensor(np.array(returns)).to(DEVICE)
        advantages = torch.FloatTensor(np.array(advantages)).to(DEVICE)
        
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        
        for _ in range(self.config['ppo_epochs']):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.config['batch_size']):
                end = start + self.config['batch_size']
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                dist = self.policy.get_distribution(batch_states)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                ratio = (new_log_probs - batch_old_log_probs).exp()
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config['clip_ratio'], 
                                   1 + self.config['clip_ratio']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_pred = self.value_net(batch_states)
                value_loss = (value_pred - batch_returns).pow(2).mean()
                
                entropy = dist.entropy().mean()
                
                total_loss = (policy_loss + 
                            self.config['value_coef'] * value_loss - 
                            self.config['entropy_coef'] * entropy)
                
                self.policy_optim.zero_grad()
                self.value_optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.policy_optim.step()
                self.value_optim.step()
    
    def train(self):
        for iteration in tqdm(range(self.config['num_iterations'])):
            states, actions, rewards, dones, log_probs, ep_rewards = self.collect_trajectories()
            
            states_tensor = torch.FloatTensor(np.array(states)).to(DEVICE)
            with torch.no_grad():
                values = self.value_net(states_tensor).cpu().numpy()
                
            returns, advantages = self.compute_advantages(rewards, dones, values, 
                                                        self.config['normalize_advantages'])
            
            self.update_networks(states, actions, log_probs, returns, advantages)
            
            if ep_rewards:
                avg_reward = np.mean(ep_rewards)
                self.rewards_history.append(avg_reward)
                
            if iteration % 20 == 0 and ep_rewards:
                print(f"Iteration {iteration}: Avg Reward {avg_reward:.2f}")
                
        self.env.close()
        return self.rewards_history

class ExperimentRunner:
 
    def plot_results(results, title, filename, xlabel='Iteration', ylabel='Avg Reward'):
        plt.figure(figsize=(12, 7))
        
        for label, rewards in results.items():
            color = f"C{list(results.keys()).index(label)}"

            window_size = max(20, len(rewards) // 10)  # Адаптивный размер окна
            cumsum = np.cumsum(np.insert(rewards, 0, 0)) 
            moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
            plt.plot(np.arange(window_size-1, len(rewards)), 
                    moving_avg, 
                    color=color,
                    linestyle='-',
                    linewidth=2.5,
                    label=f'{label} (avg)')

        plt.title(f"Результаты обучения\n{title}", fontsize=14, pad=20)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:  # Только если есть что отображать
            plt.legend(handles, labels, loc='upper center', 
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=2, framealpha=1.0)
        
        plt.tight_layout()
        
        os.makedirs("results", exist_ok=True)
        save_path = os.path.join("results", filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def run_steps_experiment():
        config = {
            'num_iterations': 200,
            'num_steps': 2048,
            'batch_size': 64,
            'gamma': 0.99,
            'clip_ratio': 0.2,
            'ppo_epochs': 10,
            'lr': 3e-4,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'normalize_advantages': True
        }
        
        steps_options = [1024, 2048, 4096]
        results = {}
        
        for steps in steps_options:
            config['num_steps'] = steps
            trainer = PPOTrainer("MountainCarContinuous-v0", config)
            results[f"Steps={steps}"] = trainer.train()
            
        ExperimentRunner.plot_results(results, "Влияние длины траектории", "steps_experiment.png")
    
    def run_clip_experiment():
        config = {
            'num_iterations': 200,
            'num_steps': 2048,
            'batch_size': 64,
            'gamma': 0.99,
            'clip_ratio': 0.2,
            'ppo_epochs': 10,
            'lr': 3e-4,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'normalize_advantages': True
        }
        
        clip_options = [0.1, 0.2, 0.3]
        results = {}
        
        for clip in clip_options:
            config['clip_ratio'] = clip
            trainer = PPOTrainer("MountainCarContinuous-v0", config)
            results[f"Clip={clip}"] = trainer.train()
            
        ExperimentRunner.plot_results(results, "Влияние коэффициента обрезки", "clip_experiment.png")
    
    def run_epochs_experiment():
        config = {
            'num_iterations': 200,
            'num_steps': 2048,
            'batch_size': 64,
            'gamma': 0.99,
            'clip_ratio': 0.2,
            'ppo_epochs': 10,
            'lr': 3e-4,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'normalize_advantages': True
        }
        
        epochs_options = [5, 10, 20]
        results = {}
        
        for epochs in epochs_options:
            config['ppo_epochs'] = epochs
            trainer = PPOTrainer("MountainCarContinuous-v0", config)
            results[f"Epochs={epochs}"] = trainer.train()
            
        ExperimentRunner.plot_results(results, "Влияние количества эпох PPO", "epochs_experiment.png")

    def run_norm_experiment():
        config = {
            'num_iterations': 200,
            'num_steps': 2048,
            'batch_size': 64,
            'gamma': 0.99,
            'clip_ratio': 0.2,
            'ppo_epochs': 10,
            'lr': 3e-4,
            'value_coef': 0.5,
            'entropy_coef': 0.01
        }
        
        results = {}
        
        print("\nЗапуск с нормализацией преимуществ...")
        config['normalize_advantages'] = True
        trainer = PPOTrainer("MountainCarContinuous-v0", config)
        results["С нормализацией"] = trainer.train()
        
        print("\nЗапуск без нормализации преимуществ...")
        config['normalize_advantages'] = False
        trainer = PPOTrainer("MountainCarContinuous-v0", config)
        results["Без нормализации"] = trainer.train()
        
        ExperimentRunner.plot_results(results, 
                                   "Сравнение нормализации преимуществ", 
                                   "norm_comparison.png")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    print("=== Эксперимент 1: Длина траектории ===")
    ExperimentRunner.run_steps_experiment()
    
    print("\n=== Эксперимент 2: Коэффициент обрезки ===")
    ExperimentRunner.run_clip_experiment()
    
    print("\n=== Эксперимент 3: Количество эпох ===")
    ExperimentRunner.run_epochs_experiment()

    print("\n=== Эксперимент 4: Нормализация преимуществ ===")
    ExperimentRunner.run_norm_experiment()