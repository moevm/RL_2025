import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent


def run_experiment(steps, clip_ratio, n_epoch, with_normalization, seed=42):
    env = gym.make("MountainCarContinuous-v0")
    torch.manual_seed(seed)
    env.reset(seed=seed)
    np.random.seed(seed)
    agent = Agent(
        env=env,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coefficient=0.4,
        clip_ratio=clip_ratio,
        steps=steps,
        iterations=300,
        n_epoch=n_epoch,
        batch_size=32,
        lr=3e-4,
        with_normalization=with_normalization
    )
    agent.train()
    return agent.scores


def run_and_plot(param_values, param_name, train_kwargs, filename_prefix):
    plt.figure(figsize=(10, 6))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, val in enumerate(param_values):
        print(f"Обучение при {param_name} = {val}")
        kwargs = train_kwargs(val)
        scores = run_experiment(
            steps=kwargs.get("steps", 2048),
            clip_ratio=kwargs.get("clip_ratio", 0.2),
            n_epoch=kwargs.get("n_epoch", 10),
            with_normalization=kwargs.get("with_normalization", True)
        )

        color = color_cycle[i % len(color_cycle)]

        plt.plot(range(len(scores)), scores, linestyle='--', alpha=0.3,
                 color=color, label=f"{param_name}={val} (raw)")

        smoothed = np.convolve(scores, np.ones(10) / 10, mode='valid')
        plt.plot(range(len(smoothed)), smoothed, linestyle='-', color=color,
                 label=f"{param_name}={val} (smoothed)")

    plt.title(f"PPO: сравнение по параметру {param_name}")
    plt.xlabel("Итерация")
    plt.ylabel("Счёт")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{param_name}.png")


def different_steps():
    steps = [1024, 2048, 4096]
    run_and_plot(steps, "steps", lambda st: {"steps": st}, "results")


def different_clip_ratio():
    clip_ratio = [0.1, 0.2, 0.3]
    run_and_plot(clip_ratio, "clip_ratio", lambda cr: {"clip_ratio": cr}, "results")


def different_normalization():
    with_normalization = [False, True]
    run_and_plot(with_normalization, "with_normalization", lambda wn: {"with_normalization": wn}, "results")


def different_n_epoch():
    n_epoch = [5, 10, 20]
    run_and_plot(n_epoch, "n_epoch", lambda ne: {"n_epoch": ne}, "results")


def main():
    different_steps()
    different_clip_ratio()
    different_normalization()
    different_n_epoch()


if __name__ == "__main__":
    main()
