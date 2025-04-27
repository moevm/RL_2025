import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import math

import random
import torch
from torch import nn
import yaml

from replay_memory import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import os
import time

from logger import Logger

DATE_FORMAT = "%m-%d %H:%M:%S"
TIME_FORMAT = "%H:%M:%S"
RUNS_DIR = "run_info"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent():
    def __init__(self):
        with open('hyperparameters.yml', 'r') as file:
            hyperparameters = yaml.safe_load(file)

        self.env_id             = hyperparameters['env_id']
        self.alpha              = hyperparameters['alpha']        # learning rate (alpha)
        self.gamma              = hyperparameters['gamma']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.train_episodes     = hyperparameters['train_episodes']
        self.test_episodes      = hyperparameters['test_episodes']
        self.network_size       = 'normal'

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE   = os.path.join(RUNS_DIR, 'logs.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, 'model.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, 'reward_graph.png')
    

    def learn(self):
        time_now = time.time()
        logger = Logger(self.LOG_FILE)
        logger.info(f"{datetime.now().strftime(DATE_FORMAT)}: Начало тренировки")
        
        env = gym.make(self.env_id, render_mode=None)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        rewards_per_episode = []
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.network_size).to(device)

        epsilon = self.epsilon_init
        memory = ReplayMemory(self.replay_memory_size)
        target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.network_size).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.alpha)
        step_count = 0
        best_reward = -9999999

        for episode in range(self.train_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            done = False
            episode_reward = 0.0

            while(not done):
                sample = random.random()
                epsilon_threshold = self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(-1.0 * step_count / self.epsilon_decay)

                if sample < epsilon_threshold:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, done, truncated, info = env.step(action.item())
                episode_reward += reward
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                memory.append((state, action, new_state, reward, done))
                step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)

            if episode_reward > best_reward:
                logger.info(f"{datetime.now().strftime(DATE_FORMAT)}: Награда достигла значения {episode_reward} на эпизоде {episode}, модель сохраняется")
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward = episode_reward

            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
            
            if episode_reward >= self.stop_on_reward:
                break
        
        time_now = round(time.time() - time_now, 2)
        logger.info(f"{datetime.now().strftime(DATE_FORMAT)}: Время обучения: {time_now} секунд")
        self.save_graph(rewards_per_episode)
    

    def test(self):
        env = gym.make(self.env_id, render_mode='human')
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.network_size).to(device)
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
        policy_dqn.eval()

        for episode in range(self.test_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            done = False

            while(not done):
                with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, done, truncated, info = env.step(action.item())
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                state = new_state


    def save_graph(self, rewards_per_episode):
        fig = plt.figure(1)

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards_per_episode)
        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.gamma * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dqn = Agent()

    if args.train:
        dqn.learn()
    else:
        dqn.test()