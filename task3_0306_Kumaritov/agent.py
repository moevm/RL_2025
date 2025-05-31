from itertools import count

import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from nets import Actor, Critic_DoubleQ
from replay_memory import ReplayMemory


class SAC():
    def __init__(self, alpha=0.4, is_auto_alpha=False, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env = self.env = gym.make("FlappyBird-v0", use_lidar=False)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        self.n_actions = self.env.action_space.n
        state, _ = self.env.reset(seed=seed)
        self.n_observations = len(state)

        self.num_episodes = 1000
        self.tau = 0.005
        self.gamma = 0.99
        self.lr = 1e-4
        self.hidden_size = 256
        self.alpha = alpha
        self.is_auto_alpha = is_auto_alpha
        self.replay_memory_size = 10000
        self.memory = ReplayMemory(self.replay_memory_size)
        self.batch_size = 256

        self.actor = Actor(self.n_observations, self.n_actions, self.hidden_size)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.lr, amsgrad=True)

        self.critic = Critic_DoubleQ(self.n_observations, self.n_actions, self.hidden_size)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.lr, amsgrad=True)

        self.critic_target = Critic_DoubleQ(self.n_observations, self.n_actions, self.hidden_size)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.is_auto_alpha:
            self.target_entropy = 0.6 * (-np.log(1 / self.n_actions))
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float32, requires_grad=True)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=self.lr, amsgrad=True)

        self.steps_done = 0

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :])
            probs = self.actor(state)
            action = torch.multinomial(probs, num_samples=1).item()
            return action

    def update_critic(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(batch[1]).unsqueeze(1)

        non_final_mask = torch.tensor([s is not None for s in batch[2]], dtype=torch.bool)
        non_final_next_states = [s for s in batch[2] if s is not None]
        if non_final_next_states:
            next_states = torch.FloatTensor(np.array(non_final_next_states))
        else:
            next_states = torch.empty((0, self.n_observations), dtype=torch.float32)

        rewards = torch.FloatTensor(batch[3]).unsqueeze(1)

        with torch.no_grad():
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)

            next_q1, next_q2 = self.critic_target(next_states)
            min_q = torch.min(next_q1, next_q2)

            next_v = (next_probs * (min_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards.clone()
            target_q[non_final_mask] += self.gamma * next_v

        q1, q2 = self.critic(states)
        q1_pred = q1.gather(1, actions)
        q2_pred = q2.gather(1, actions)
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self):
        states = torch.FloatTensor(
            np.array([t[0] for t in self.memory.sample(self.batch_size)]))

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-10)

        q1, q2 = self.critic(states)
        min_q = torch.min(q1, q2)

        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.is_auto_alpha:
            entropy = - (probs * log_probs).sum(dim=1).mean()
            alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = min(max(self.log_alpha.exp().item(), 0.001), 1.0)

    def train(self):
        self.actor.train()
        self.critic.train()
        self.critic_target.train()

        episode_rewards = []
        episode_alphas = []
        log_interval = max(1, self.num_episodes // 10)
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            for _ in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                done = terminated or truncated

                if done:
                    next_state = None
                else:
                    next_state = observation

                self.memory.push(state, action, next_state, reward)
                state = next_state

                if len(self.memory) >= self.batch_size:
                    self.update_critic()
                    self.update_actor()

                    for target, policy in zip(self.critic_target.parameters(), self.critic.parameters()):
                        target.data.copy_(self.tau * policy.data + (1 - self.tau) * target.data)

                if done:
                    episode_alphas.append(self.alpha)
                    episode_rewards.append(total_reward)
                    break

            if episode % log_interval == 0:
                percent_done = (episode / self.num_episodes) * 100
                avg_reward = np.mean(episode_rewards[-log_interval:])
                print(
                    f"Training progress: {percent_done:.0f}% ({episode}/{self.num_episodes} episodes), average reward {avg_reward:.2f}")

        self.env.close()
        self.memory.clear()
        return episode_rewards, episode_alphas
