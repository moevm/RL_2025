import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
from collections import deque
from torch import nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import os
import itertools

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

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
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions, architecture='default'):
        super().__init__()
        self.architecture = architecture
        
        if architecture == 'default':
            self.net = nn.Sequential(
                nn.Linear(obs_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions)
            )
        elif architecture == 'small':
            self.net = nn.Sequential(
                nn.Linear(obs_size, 32),
                nn.ReLU(),
                nn.Linear(32, n_actions)
            )
        elif architecture == 'large':
            self.net = nn.Sequential(
                nn.Linear(obs_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions)
            )
        elif architecture == 'deep':
            self.net = nn.Sequential(
                nn.Linear(obs_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions)
            )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_size, n_actions, architecture='default', 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.995, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.q_net = QNetwork(obs_size, n_actions, architecture).to(self.device)
        self.target_net = QNetwork(obs_size, n_actions, architecture).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        
        self.gamma = gamma
        self.batch_size = 128
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.target_update_freq = 10
        
        self.replay_buffer = ReplayBuffer(10000)
        self.steps_done = 0
    
    def select_action(self, state):
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q_net(state_tensor)
                return q_values.argmax().item()
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        current_q = self.q_net(state).gather(1, action.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0]
            target_q = reward + (1 - done) * self.gamma * next_q
        
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

def plot_results(all_rewards, title, window=100):
    plt.figure(figsize=(15, 6))
    
    # Raw rewards
    plt.subplot(1, 2, 1)
    for label, rewards in all_rewards.items():
        plt.plot(rewards, alpha=0.3, label=f'{label} raw')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Raw {title}')
    plt.legend()
    
    # Rolling mean
    plt.subplot(1, 2, 2)
    for label, rewards in all_rewards.items():
        rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(rolling_mean, label=f'{label} mean')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Rolling Mean (window={window})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()

def train_agent(architecture='default', episodes=500, gamma=0.99, 
                epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    os.makedirs('./videos', exist_ok=True)
    
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    video_folder = f"./videos/{architecture}_g{gamma}_d{epsilon_decay}_e{epsilon_start}"
    env = RecordVideo(env, video_folder=video_folder, fps=20, episode_trigger=lambda x: x % 50 == 0)
    
    agent = DQNAgent(
        obs_size=4, 
        n_actions=2, 
        architecture=architecture,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay
    )
    
    reward_history = []
    best_mean_reward = -np.inf
    
    for episode in tqdm(range(episodes), 
                       desc=f"Training {architecture} (γ={gamma}, d={epsilon_decay}, ε₀={epsilon_start})"):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            agent.train()
            
            if done:
                break
        
        if episode % agent.target_update_freq == 0:
            agent.update_target()
        
        reward_history.append(episode_reward)
        
        if episode % 50 == 0:
            mean_reward = np.mean(reward_history[-50:])
            print(f"{architecture} (γ={gamma}, d={epsilon_decay}, ε₀={epsilon_start}) | "
                  f"Episode {episode}: Mean Reward={mean_reward:.1f}, Epsilon={agent.epsilon:.3f}")
            
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                model_name = f'best_{architecture}_g{gamma}_d{epsilon_decay}_e{epsilon_start}_model.pth'
                torch.save(agent.q_net.state_dict(), model_name)
    
    env.close()
    return reward_history

def compare_architectures():
    architectures = ['default', 'small', 'large', 'deep']
    all_rewards = {}
    
    for arch in architectures:
        rewards = train_agent(architecture=arch, episodes=300)
        all_rewards[arch] = rewards
        
        plt.figure()
        plt.plot(rewards)
        plt.title(f'Training Progress - {arch} architecture')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(f'{arch}_training.png')
        plt.close()
    
    plot_results(all_rewards, 'Architecture Comparison')
    
    print("\nArchitecture Comparison Results:")
    print("Architecture | Mean Reward (last 50) | Stability | Best Reward")
    print("------------------------------------------------------------")
    for arch, rewards in all_rewards.items():
        last_50 = rewards[-50:]
        print(f"{arch:10} | {np.mean(last_50):19.1f} | {np.std(last_50):8.2f} | {max(rewards):11}")

def compare_gammas():
    gammas = [0.9, 0.95, 0.99, 0.999]
    all_rewards = {}
    
    for gamma in gammas:
        rewards = train_agent(architecture='default', episodes=300, gamma=gamma)
        all_rewards[f'γ={gamma}'] = rewards
    
    plot_results(all_rewards, 'Gamma Comparison')
    
    print("\nGamma Comparison Results:")
    print("Gamma | Mean Reward (last 50) | Stability | Best Reward")
    print("------------------------------------------------------")
    for label, rewards in all_rewards.items():
        last_50 = rewards[-50:]
        print(f"{label:5} | {np.mean(last_50):19.1f} | {np.std(last_50):8.2f} | {max(rewards):11}")

def compare_epsilon_decays():
    epsilon_decays = [0.99, 0.995, 0.999]
    all_rewards = {}
    
    for decay in epsilon_decays:
        rewards = train_agent(architecture='default', episodes=300, epsilon_decay=decay)
        all_rewards[f'd={decay}'] = rewards
    
    plot_results(all_rewards, 'Epsilon Decay Comparison')
    
    print("\nEpsilon Decay Comparison Results:")
    print("Decay | Mean Reward (last 50) | Stability | Best Reward")
    print("------------------------------------------------------")
    for label, rewards in all_rewards.items():
        last_50 = rewards[-50:]
        print(f"{label:5} | {np.mean(last_50):19.1f} | {np.std(last_50):8.2f} | {max(rewards):11}")

def compare_epsilon_starts():
    epsilon_starts = [0.5, 1.0, 1.5]
    all_rewards = {}
    
    for start in epsilon_starts:
        rewards = train_agent(architecture='default', episodes=300, epsilon_start=start)
        all_rewards[f'ε₀={start}'] = rewards
    
    plot_results(all_rewards, 'Epsilon Start Comparison')
    
    print("\nEpsilon Start Comparison Results:")
    print("Start | Mean Reward (last 50) | Stability | Best Reward")
    print("------------------------------------------------------")
    for label, rewards in all_rewards.items():
        last_50 = rewards[-50:]
        print(f"{label:5} | {np.mean(last_50):19.1f} | {np.std(last_50):8.2f} | {max(rewards):11}")

def run_full_experiment():
    print("=== Architecture Comparison ===")
    compare_architectures()
    
    print("\n=== Gamma Comparison ===")
    compare_gammas()
    
    print("\n=== Epsilon Decay Comparison ===")
    compare_epsilon_decays()
    
    print("\n=== Epsilon Start Comparison ===")
    compare_epsilon_starts()

if __name__ == "__main__":
    run_full_experiment()