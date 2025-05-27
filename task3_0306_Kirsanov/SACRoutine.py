import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import gymnasium as gym
import flappy_bird_gymnasium
from gymnasium.wrappers import RecordVideo

from collections import deque

import random
import os
import time

import numpy as np

import matplotlib.pyplot as plt

import argparse

# Класс для ReplayBuffer
class ReplayBuffer:
  def __init__(self, capacity=1000):
      self.buffer = deque(maxlen=capacity)

  def push(self, observation, next_observation, action, reward, termination):
      self.buffer.append((observation, next_observation, action, reward,  termination))

  def sample(self, batch_size):
      batch = random.sample(self.buffer, batch_size)
      observations, next_observations, actions, rewards, terminations = zip(*batch)
      return (
         #torch.tensor(np.array(state), dtype=torch.float32),
          torch.tensor(np.array(observations), dtype=torch.float32),
          torch.tensor(np.array(next_observations), dtype=torch.float32),
          torch.tensor(np.array(actions), dtype=torch.float32),
          torch.tensor(np.array(rewards), dtype=torch.float32),
          torch.tensor(np.array(terminations), dtype=torch.float32)
      )

  def __len__(self):
      return len(self.buffer)

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

#Класс Q-сети
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim))
        )

    def forward(self, x):
        q_vals = self.net(x)
        return q_vals

#Класс Actor-сети
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hiddenSize=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hiddenSize)),
            nn.ReLU(),
            layer_init(nn.Linear(hiddenSize, hiddenSize)),
            nn.ReLU(),
            layer_init(nn.Linear(hiddenSize, hiddenSize)),
            nn.ReLU(),
            layer_init(nn.Linear(hiddenSize, action_dim))
        )

    def forward(self, x):
        logits = self.net(x) #предсказание сети актора
        return logits

    def get_dist(self, state):
        logits = self.forward(state)
        return Categorical(logits=logits)

    def get_action(self, state):
        #Вариант с которым не работало
        # #state = torch.FloatTensor(state).to(device)
        # dist = self.get_dist(state)
        # action = dist.sample()
        # return action, dist.log_prob(action), dist.probs

        #Перетащил пробы с sac_atari
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        actionProbs = dist.probs
        # dim = 0 потому что данные не с нескольких сред -> предположение
        logProb = F.log_softmax(logits, dim=0)
        return action, logProb, actionProbs


# Парсинг аргументов
parser = argparse.ArgumentParser(description='SAC for Flappy')
#Параметры среды
parser.add_argument('--LIDAR', type=bool, default=True, help='If true use LIDAR as observation_space')
parser.add_argument('--seed', type=int, default=69, help='seed of the experiment')
parser.add_argument('--cuda', type=bool, default=True, help='if toggled, cuda will be enabled by default')
parser.add_argument('--video', type=bool, default=False, help='whether to capture videos of the agent performances (check out `videos` folder)')
parser.add_argument('--steps_for_statistic', type=int, default=8000, help='steps for print statistic')



#Параметры алгоритма
parser.add_argument('--total_timesteps', type=int, default=5000000, help='total timesteps of the experiments')
parser.add_argument('--buffer_size', type=int, default=int(1e6), help='the replay memory buffer size')
parser.add_argument('--batch_size', type=int, default=64, help='the batch size of sample from the reply memory')
parser.add_argument('--update_frequency', type=int, default=4, help='the frequency of training updates')
parser.add_argument('--target_network_frequency', type=int, default=8000, help='the frequency of updates for the target networks')
parser.add_argument('--learning_starts', type=int, default=2e4, help='timestep to start learning')

##Параметры сеток
parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor gamma')
parser.add_argument('--tau', type=float, default=1.0, help='target smoothing coefficient (default: 1)')
parser.add_argument('--policy_lr', type=float, default=3e-4, help='the learning rate of the policy network optimizer')
parser.add_argument('--q_lr', type=float, default=3e-4, help='the learning rate of the Q network network optimizer')
#Параметры автотюна alpha
parser.add_argument('--alpha', type=float, default=0.2, help='Entropy regularization coefficient')
parser.add_argument('--autotune', type=bool, default=True, help='automatic tuning of the entropy coefficient')
parser.add_argument('--target-entropy-scale', type=float, default=0.89, help='coefficient for scaling the autotune entropy target')
args = parser.parse_args()




if __name__ == "__main__":
    # Девайс
    if(args.cuda):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Инициализация окружения
    #env = gym.make("FlappyBird-v0", render_mode='human', use_lidar=args.LIDAR)
    env = gym.make("FlappyBird-v0", render_mode='rgb_array', use_lidar=args.LIDAR)

    if (args.video):
        env = RecordVideo(
            env,
            video_folder="./videos",
            ###Посмотреть что ставить
            episode_trigger=lambda t: t % 1-0 == 0,  # Record every episode
            video_length=100,
            name_prefix="FlappyBird-v0"
        )


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Размеры пространства состояний и действий
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim).to(device)
    qf1 = SoftQNetwork(state_dim, action_dim).to(device)
    qf2 = SoftQNetwork(state_dim, action_dim).to(device)
    qf1_target = SoftQNetwork(state_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(state_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.q_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(action_dim))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(args.buffer_size)

    #Для вывода жизни агента
    start_time = time.time()
    #Накопление числа перезапусков
    reloads_list = []
    reloads_counter = 0
    #Цикл обучения
    obs, _ = env.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action, _, _ = actor.get_action(torch.tensor(obs, device=device, dtype=torch.float32))
            action = action.detach().cpu().numpy()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, _, _ = env.step(action)

        if (termination):
            reloads_counter += 1

        rb.push(obs, next_obs, action, reward, termination)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                observations, next_observations, actions, rewards, terminations = rb.sample(args.batch_size)
                observations = observations.to(device)
                next_observations = next_observations.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                terminations = terminations.to(device)

                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(next_observations)
                    qf1_next_target = qf1_target(next_observations)
                    qf2_next_target = qf2_target(next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                            torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = rewards.flatten() + (1 - terminations.flatten()) * args.gamma * (
                        min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(observations)
                qf2_values = qf2(observations)
                #gather позаимствовал с моего DQN
                qf1_a_values = qf1_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
                qf2_a_values = qf2_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(observations)
                with torch.no_grad():
                    qf1_values = qf1(observations)
                    qf2_values = qf2(observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)

                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % args.steps_for_statistic == 0:
                print("SPS:", int(global_step / (time.time() - start_time)), "Reloads: ", reloads_counter)
                reloads_list.append(reloads_counter)
                reloads_counter = 0

    env.close()