import numpy as np
import torch
import torch.optim as optim

from actor import Actor
from critic import Critic


class Agent:
    def __init__(
            self,
            env,
            gamma,
            gae_lambda,
            entropy_coefficient,
            clip_ratio,
            steps,
            iterations,
            n_epoch,
            batch_size,
            lr,
            with_normalization
    ):
        self.env = env
        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient
        self.clip_ratio = clip_ratio
        self.steps = steps
        self.iterations = iterations
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.with_normalization = with_normalization

        self.actor = Actor(self.n_observations, self.n_actions)
        self.critic = Critic(self.n_observations)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.actor_loss_history = []
        self.critic_loss_history = []
        self.scores = []

    def get_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        action, _ = self.actor.get_action(state)
        return action.item()

    def make_action(self, action):
        clipped_action = np.clip(action, -1, 1)
        next_state, reward, terminated, truncated, _ = self.env.step(clipped_action)
        is_terminal = terminated or truncated
        return next_state, reward, is_terminal

    def train(self):
        for _ in range(self.iterations):
            states, actions, rewards, values, is_terminals, log_probs, episode_rewards = self.get_trajectories()
            returns, advantages = self.get_returns_and_advantages(rewards, values, is_terminals)
            self.update_net_weights(states, actions, log_probs, returns, advantages)

            if episode_rewards:
                avg_score = np.mean(episode_rewards)
                self.scores.append(avg_score)

        self.env.close()

    def update_net_weights(self, states, actions, old_log_probs, returns, advantages):
        self.actor.train()
        self.critic.train()

        actor_losses, critic_losses = [], []

        states = np.array(states)
        actions = np.array(actions)
        old_log_probs = np.array(old_log_probs)

        for _ in range(self.n_epoch):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = torch.as_tensor(states[batch_indices], dtype=torch.float32)
                batch_actions = torch.as_tensor(actions[batch_indices], dtype=torch.float32)
                batch_old_log_probs = torch.as_tensor(old_log_probs[batch_indices], dtype=torch.float32)
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                dist = self.actor.get_dist(batch_states)
                cur_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                ratio = torch.exp(cur_log_probs - batch_old_log_probs)
                entropy = dist.entropy().mean()
                batch_advantages = batch_advantages.detach()
                loss = batch_advantages * ratio
                clipped_loss = (
                        torch.clamp(ratio, 1. - self.clip_ratio, 1. + self.clip_ratio)
                        * batch_advantages
                )
                actor_loss = (
                        -torch.mean(torch.min(loss, clipped_loss))
                        - entropy * self.entropy_coefficient
                )
                cur_value = self.critic(batch_states)
                critic_loss = (batch_returns - cur_value).pow(2).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        avg_actor_loss = sum(actor_losses) / len(actor_losses)
        avg_critic_loss = sum(critic_losses) / len(critic_losses)
        self.actor_loss_history.append(avg_actor_loss)
        self.critic_loss_history.append(avg_critic_loss)

    def get_returns_and_advantages(self, rewards, values, is_terminals):
        gae = 0
        returns, advantages = [], []

        for i in reversed(range(len(rewards))):
            if is_terminals[i]:
                next_value = 0.0
            delta = rewards[i] + self.gamma * (values[i + 1] if i < len(rewards) - 1 else 0) - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            returns.insert(0, (gae + values[i]).detach().clone().float())
            advantages.insert(0, gae.detach().clone().float())

        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)

        if self.with_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def get_trajectories(self):
        states, actions, rewards, values, is_terminals, episode_rewards, log_probs = [], [], [], [], [], [], []
        current_reward = 0
        state, _ = self.env.reset()

        for _ in range(self.steps):
            state_tensor = torch.as_tensor(np.array(state), dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_tensor, log_prob = self.actor.get_action(state_tensor)
                value = self.critic(state_tensor)

            action = np.array([action_tensor.item()])
            next_state, reward, is_terminal = self.make_action(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            is_terminals.append(is_terminal)
            values.append(value.squeeze())
            current_reward += reward
            state = next_state

            if is_terminal:
                episode_rewards.append(current_reward)
                current_reward = 0
                state, _ = self.env.reset()

        return states, actions, rewards, values, is_terminals, log_probs, episode_rewards
