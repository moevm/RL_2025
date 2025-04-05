import math
import random
from collections import namedtuple, deque
from itertools import count
from typing import Type

import gym
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Гиперпараметры.
BATCH_SIZE = 128  # размер мини-батча.
GAMMA = 0.99  # коэффициент дисконтирования.
EPS_START = 0.9  # начальное значение ε для ε-жадной стратегии.
EPS_END = 0.05  # минимальное значение ε.
EPS_DECAY = (
    500  # скорость уменьшения ε (чем меньше, тем быстрее убывает, но должна быть > 0).
)
TAU = 0.005  # скорость обновления target нейронной сети.
LR = 1e-4  # скорость обучения.

# Определяем структуру для хранения переходов (experience tuple).
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


# Реализация буфера воспоминаний (replay memory).
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Сохраняем переход"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Случайным образом выбираем батч переходов"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Базовая нейронная сеть для аппроксимации Q-функции.
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


# Нейронная сеть поменьше для аппроксимации Q-функции.
class DQN_small(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN_small, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, n_actions)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


# Нейронная сеть побольше для аппроксимации Q-функции.
class DQN_big(nn.Module):
    def __init__(self, n_observations, n_actions, p=0.4):
        super(DQN_big, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 512)
        self.layer4 = nn.Linear(512, 256)
        self.layer5 = nn.Linear(256, 128)
        self.layer6 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        return self.layer6(x)


# Глобальная переменная для отслеживания числа шагов (для расчёта ε).
steps_done = 0


def select_action(state: list[float], policy_net: nn.Module, n_actions: int):
    """
    Выбирает действие с использованием ε‑жадной стратегии.
    С вероятностью (1-ε) выбирается действие с максимальным Q,
    иначе – случайное действие.
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # Выбираем действие с максимальным Q-значением.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # Случайное действие.
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


def optimize_model(
    policy_net: nn.Module,
    target_net: nn.Module,
    memory: ReplayMemory,
    optimizer: optim.Adam,
):
    """
    Функция оптимизации модели на основе мини-батча из replay memory.
    """
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Преобразуем список переходов в батч.
    batch = Transition(*zip(*transitions))

    # Создаем маску для тех переходов, где следующее состояние не None.
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Вычисляем Q(s, a) для текущих состояний с помощью policy-сети.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Вычисляем максимальные Q-значения для следующих состояний с использованием target-сети.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Вычисляем целевые значения Q: r + γ * max Q(next_state, a).
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze()

    # Рассчитываем функцию потерь (MSE).
    loss = nn.SmoothL1Loss()(
        state_action_values.squeeze(), expected_state_action_values
    )

    # Обновляем веса сети.
    optimizer.zero_grad()
    loss.backward()
    # Ограничиваем градиенты для стабильности обучения.
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def plot_durations(episodes_durations: list[list[float]], episodes_legends: list[str]):
    plt.figure(1)

    for i, episode_durations in enumerate(episodes_durations):

        durations_t = torch.tensor(episode_durations, dtype=torch.float)

        plt.title("Результат")
        plt.xlabel("Номер эпизода")
        plt.ylabel("Колличество шагов в эпизоде (max=500)")
        plt.plot(
            durations_t.numpy(), alpha=0.5, label=episodes_legends[i], color=f"C{i}"
        )

        # Усредняем по 100 эпизодам и тоже выводим.
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), color=f"C{i}")

    plt.legend(loc="best")
    plt.show()


def train_DQN(
    env: gym.Env, nn_model: Type[nn.Module], num_episodes=600, buff_size=10000
):
    """
    Запускает 1 полное обучение DQN.

    :param env: созданная среда.
    :param nn_model: класс архитектуры нейронной сети, которая будет обучаться.
    :param num_episodes: количество эпизодов обучения.
    :param buff_size: размер ReplayMemory буфера.

    :return episode_durations: массив, который содержит длительность для каждой запущенной симуляции.
    """
    global steps_done
    steps_done = 0

    # Получаем размер наблюдения и количество действий из среды.
    initial_state, _ = env.reset()
    n_observations = len(initial_state)
    n_actions = env.action_space.n

    # Инициализируем policy-сеть и target-сеть.
    policy_net = nn_model(n_observations, n_actions).to(device)
    target_net = nn_model(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(buff_size)

    episode_durations = []

    episodes_tqdm = tqdm(range(num_episodes))
    for i_episode in episodes_tqdm:
        # Инициализируем состояние среды.
        state, _ = env.reset()
        state = torch.tensor([state], device=device, dtype=torch.float32)

        for t in count():
            # Выбираем действие на основе текущего состояния.
            action = select_action(state, policy_net, n_actions)
            # Выполняем действие в среде.
            next_state, reward, done, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            if done or truncated:
                next_state_tensor = None
            else:
                next_state_tensor = torch.tensor(
                    [next_state], device=device, dtype=torch.float32
                )

            # Сохраняем переход в replay memory.
            memory.push(state, action, next_state_tensor, reward)

            # Переходим к следующему состоянию.
            state = next_state_tensor if next_state_tensor is not None else None

            optimize_model(policy_net, target_net, memory, optimizer)

            # Мягкое обновление весов target net.
            # θ′ ← τ*θ + (1−τ)*θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done or truncated:
                episode_durations.append(t + 1)
                episodes_tqdm.desc = f"Последний эпизод продолжался :({t+1} шаг)"
                break

    print("Обучение завершено")
    return episode_durations


if __name__ == "__main__":
    episodes_durations = []
    env = gym.make("CartPole-v1")

    # Изменение параметров для эксперимента.
    params = [(env, DQN), (env, DQN), (env, DQN), (env, DQN)]
    eps_start_experiment = [0.2, 0.5, 0.7, 0.9]
    episodes_legends = [
        "eps_start = 0.2",
        "eps_start = 0.5",
        "eps_start = 0.7",
        "eps_start = 0.9",
    ]

    for i, param in enumerate(params):
        # Меняем значение гиперпараметра, если эксперимент заключается в этом.
        EPS_START = eps_start_experiment[i]

        episode_durations = train_DQN(*param)

        episodes_durations.append(episode_durations)

    plot_durations(episodes_durations, episodes_legends)
