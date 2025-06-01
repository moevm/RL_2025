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
from tqdm import tqdm

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MINI_BATCH = 128

class NetDesigns:
    @staticmethod
    def compact(in_dim, out_dim):
        features = 32
        return nn.Sequential(
            nn.Linear(in_dim, features),
            nn.ReLU(),
            nn.Linear(features, out_dim))

    @staticmethod
    def balanced(in_dim, out_dim):
        features = 64
        return nn.Sequential(
            nn.Linear(in_dim, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, out_dim))

    @staticmethod
    def expanded(in_dim, out_dim):
        features = 128
        features_2 = 64
        return nn.Sequential(
            nn.Linear(in_dim, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features_2),
            nn.ReLU(),
            nn.Linear(features_2, out_dim))

class MemoryBuffer:
    def __init__(self, capacity=10000):
        self.data = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def add(self, state, action, reward, next_state, done):
        self.data.append((state, action, reward, next_state, done))

    def get_batch(self, size):
        samples = random.sample(self.data, size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.FloatTensor(np.array(states)).to(DEV),
            torch.LongTensor(np.array(actions)).to(DEV),
            torch.FloatTensor(np.array(rewards)).to(DEV),
            torch.FloatTensor(np.array(next_states)).to(DEV),
            torch.FloatTensor(np.array(dones)).to(DEV))


class QLearningAgent:
    def __init__(self, obs_dim, action_count,
                 net_type='balanced',
                 discount=0.99,
                 eps_start=1.0,
                 eps_min=0.01,
                 eps_decay=0.995,
                 learn_rate=1e-4,
                 blend=0.005):

        self.online_net = getattr(NetDesigns, net_type)(obs_dim, action_count).to(DEV)
        self.target_net = getattr(NetDesigns, net_type)(obs_dim, action_count).to(DEV)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.opt = optim.Adam(self.online_net.parameters(), lr=learn_rate)
        self.memory = MemoryBuffer()

        self.discount = discount
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.blend = blend
        self.action_count = action_count

        self.episode_lengths = []
        self.total_steps = 0

    def choose_action(self, state):
        self.total_steps += 1
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

        if random.random() < self.eps:
            return random.randint(0, self.action_count - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEV)
            return self.online_net(state_tensor).argmax().item()

    def optimize(self):
        if len(self.memory) < MINI_BATCH:
            return

        s, a, r, ns, d = self.memory.get_batch(MINI_BATCH)

        q_current = self.online_net(s).gather(1, a.unsqueeze(1))
        q_next = self.target_net(ns).max(1)[0].detach()
        q_target = r + (1 - d) * self.discount * q_next

        loss = nn.MSELoss()(q_current.squeeze(), q_target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 100)
        self.opt.step()

        for t_param, o_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_param.data.copy_(self.blend * o_param.data + (1 - self.blend) * t_param.data)

    def learn(self, env, episodes=800, max_iters=500, log_every=50):
        progress_bar = tqdm(range(episodes), desc="Training Progress")

        for ep in progress_bar:
            state, _ = env.reset()
            steps = 0
            terminated = False

            while not terminated and steps < max_iters:
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                terminated = done or truncated

                self.memory.add(state, action, reward, next_state, terminated)
                self.optimize()

                state = next_state
                steps += 1

            self.episode_lengths.append(steps)

            if ep % log_every == 0:
                progress_bar.set_description(
                    f"Ep {ep}: Steps {steps}, "
                    f"ε {self.eps:.2f}"
                )

        return self.episode_lengths


class ExperimentManager:
    @staticmethod
    def test_networks(env, episodes=600):
        designs = ['compact', 'balanced', 'expanded']
        outcomes = {}

        for design in designs:
            print(f"\nTesting network: {design}")
            agent = QLearningAgent(env.observation_space.shape[0], env.action_space.n, net_type=design)
            lengths = agent.learn(env, episodes)
            outcomes[design] = lengths

        ExperimentManager._visualize(
            outcomes,
            title="Network Architecture Comparison",
            filename="net_comparison.png")

    @staticmethod
    def test_discounts(env, episodes=600):
        discounts = [0.9, 0.95, 0.99, 0.999]
        outcomes = {}

        for d in discounts:
            print(f"\nTesting discount γ={d}")
            agent = QLearningAgent(env.observation_space.shape[0], env.action_space.n, discount=d)
            lengths = agent.learn(env, episodes)
            outcomes[f"γ={d}"] = lengths

        ExperimentManager._visualize(
            outcomes,
            title="Discount Factor Comparison",
            filename="gamma_comparison.png")

    @staticmethod
    def test_epsilon_decays(env, episodes=600):
        decays = [0.99, 0.98, 0.95, 0.9]
        outcomes = {}

        for decay in decays:
            print(f"\nTesting ε decay={decay}")
            agent = QLearningAgent(
                env.observation_space.shape[0],
                env.action_space.n,
                eps_decay=decay)
            lengths = agent.learn(env, episodes)
            outcomes[f"decay={decay}"] = lengths

        ExperimentManager._visualize(
            outcomes,
            title="Epsilon Decay Comparison",
            filename="eps_decay_comparison.png")

    @staticmethod
    def test_epsilon_starts(env, episodes=600):
        starts = [0.9, 0.7, 0.5, 0.3]
        outcomes = {}

        for start in starts:
            print(f"\nTesting ε start={start}")
            agent = QLearningAgent(
                env.observation_space.shape[0],
                env.action_space.n,
                eps_start=start)
            lengths = agent.learn(env, episodes)
            outcomes[f"start={start}"] = lengths

        ExperimentManager._visualize(
            outcomes,
            title="Epsilon Start Comparison",
            filename="eps_start_comparison.png")

    @staticmethod
    def _visualize(results, title, filename):
        plt.figure(figsize=(10, 6))
        palette = plt.cm.tab10
        for idx, (label, lengths) in enumerate(results.items()):
            color = palette(idx)
            plt.plot(lengths, label=label, alpha=0.3, color=color)

            if len(lengths) >= 100:
                window = 100
                cumsum = np.cumsum(np.insert(lengths, 0, 0))
                ma = (cumsum[window:] - cumsum[:-window]) / window
                plt.plot(np.arange(window - 1, len(lengths)),
                         ma,
                         color=color,
                         linewidth=2,
                         label=f'{label} (avg)')

        plt.title(f"Episode Lengths\n{title}")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        legend_items = {}
        for handle, lbl in zip(*plt.gca().get_legend_handles_labels()):
            if '(' not in lbl:
                legend_items[lbl] = handle
        plt.legend(handles=[
            lines.Line2D([0], [0], color=h.get_color(), lw=2, label=k)
            for k, h in legend_items.items()
        ])
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("output", exist_ok=True)
        plt.savefig(f"output/{filename}", dpi=300)
        plt.close()

if __name__ == "__main__":
    environment = gym.make('CartPole-v1')
    print(f"\nNetwork Architecture Test:")
    ExperimentManager.test_networks(environment)
    print("\nDiscount Factor Test:")
    ExperimentManager.test_discounts(environment)
    print("\nEpsilon Decay Test:")
    ExperimentManager.test_epsilon_decays(environment)
    print("\nEpsilon Start Test:")
    ExperimentManager.test_epsilon_starts(environment)
    environment.close()