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


# ReplayBuffer
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
          torch.tensor(next_state, dtype=torch.float32),
          torch.tensor(done, dtype=torch.float32)
      )

  def __len__(self):
      return len(self.buffer)

# QNetwork
class QNetwork(nn.Module):

  def __init__(self, obs_size, n_actions):
      super(QNetwork, self).__init__()
      self.net = nn.Sequential(
          nn.Linear(obs_size, 96),
          nn.ReLU(),
          nn.Linear(96, 96),
          nn.ReLU(),
          nn.Linear(96, 96),
          nn.ReLU(),
          nn.Linear(96, 96),
          nn.ReLU(),
          nn.Linear(96, 96),
          nn.ReLU(),
          nn.Linear(96, n_actions),
      )

  def forward(self, x):
      return self.net(x)

# DQNAgent
class DQNAgent:

  def __init__(self, obs_size, n_actions):

      self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")

      self.q_net = QNetwork(obs_size, n_actions).to(self.device)
      self.target_net = QNetwork(obs_size, n_actions).to(self.device)
      self.target_net.load_state_dict(self.q_net.state_dict())

      self.optimizer = optim.Adam(self.q_net.parameters(), lr = 1e-3)

      self.gamma = 0.99
      self.batch_size = 64
      self.epsilon = 1
      self.epsilon_decay = 0.7
      self.epsilon_min = 0.01

      self.replay_buffer = ReplayBuffer(1000)

  def select_action(self, state):
      if random.random() < self.epsilon:
          return random.randint(0, 1)
      else:
          with torch.no_grad():
              state_tensor = torch.tensor(state, dtype=torch.float32, device= self.device)

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
      target_q_values = reward + self.gamma * self.target_net(next_state).max(1)[0] * (1 - done)

      loss = nn.MSELoss()(q_values, target_q_values)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

  def update_target(self):
      self.target_net.load_state_dict(self.q_net.state_dict())

#Work
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(
      env,
      video_folder="./videos",
      fps=20,
      video_length=1000,
      episode_trigger = lambda t: t % 10 == 0
      )

agent = DQNAgent(obs_size=4, n_actions=2)

num_episodes = 600
reward_history = []


for episode in tqdm(range(num_episodes)):
  state, _ = env.reset()
  episode_reward = 0
  for i in range(200):
      action = agent.select_action(state)
      next_state, reward, done, _, _ = env.step(action)
      agent.replay_buffer.push(state, action, reward, next_state, done)

      state = next_state
      episode_reward += reward

      agent.train()

      if done:
          break
  agent.update_target()

  agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

  reward_history.append(episode_reward)
  if ( episode + 1)  % 2 == 0:
      print(f"Episode: {episode +1}, Reward: {episode_reward}, Epsilon: {agent.epsilon}")

plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training Progress")
plt.show()


env.close