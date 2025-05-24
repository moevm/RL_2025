import gymnasium as gym
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from itertools import count
from torch import nn
from torch import optim

from dqn import DQN
from replay_memory import ReplayMemory
from transition import Transition

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.5
EPS_END = 0.05
EPS_DECAY = 500
TAU = 0.005
LR = 1e-4
num_episodes = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
env = gym.make("CartPole-v1")


def optimize_model(memory, policy_net, target_net, optimizer, gamma):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def select_action(state, policy_net, steps_done, eps_start, eps_decay):
    sample = random.random()
    eps_threshold = EPS_END + (eps_start - EPS_END) * \
                    math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1), steps_done
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), steps_done


def train_dqn(hidden_size=256, gamma=GAMMA, eps_start=EPS_START, eps_decay=EPS_DECAY):
    steps_done = 0
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions, hidden_size).to(device)
    target_net = DQN(n_observations, n_actions, hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    episode_durations = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action, steps_done = select_action(state, policy_net, steps_done, eps_start, eps_decay)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer, gamma)

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                break

    return episode_durations


def run_and_plot(param_values, param_name, train_kwargs, filename_prefix):
    plt.figure(figsize=(10, 6))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, val in enumerate(param_values):
        print(f"Обучение при {param_name} = {val}")
        steps_done = 0
        kwargs = train_kwargs(val)
        durations = train_dqn(**kwargs)

        color = color_cycle[i % len(color_cycle)]

        plt.plot(range(len(durations)), durations, linestyle='--', alpha=0.3,
                 color=color, label=f"{param_name}={val} (raw)")

        smoothed = np.convolve(durations, np.ones(10) / 10, mode='valid')
        plt.plot(range(len(smoothed)), smoothed, linestyle='-', color=color,
                 label=f"{param_name}={val} (smoothed)")

    plt.title(f"DQN: сравнение по параметру {param_name}")
    plt.xlabel("Эпизод")
    plt.ylabel("Длительность")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{param_name}.png")


def different_hidden_size():
    hidden_sizes = [64, 128, 256]
    run_and_plot(hidden_sizes, "hidden_size", lambda hz: {"hidden_size": hz}, "results")


def different_gamma():
    gammas = [0.8, 0.9, 0.99]
    run_and_plot(gammas, "gamma", lambda g: {"gamma": g}, "results")


def different_epsilon_decay():
    decays = [250, 500, 750]
    run_and_plot(decays, "eps_decay", lambda d: {"eps_decay": d}, "results")


def different_epsilon_start():
    starts = [0.25, 0.5, 0.75]
    run_and_plot(starts, "eps_start", lambda s: {"eps_start": s}, "results")


def main(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    env.reset(seed=seed)
    np.random.seed(seed)
    different_hidden_size()
    different_gamma()
    different_epsilon_decay()
    different_epsilon_start()


if __name__ == "__main__":
    main()
