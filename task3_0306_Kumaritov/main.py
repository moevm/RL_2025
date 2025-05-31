import matplotlib.pyplot as plt
import numpy as np

from agent import SAC


def run_and_plot(param_values, param_name, train_kwargs, filename_prefix):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, val in enumerate(param_values):
        print(f"Обучение в FlappyBird-v0, режим {param_name} {val}")
        kwargs = train_kwargs(val)
        sac = SAC(**kwargs, seed=42)
        episode_rewards, episode_alphas = sac.train()
        print(f"Обучение в FlappyBird-v0, режим {param_name} {val} завершено")

        color = color_cycle[i % len(color_cycle)]
        episodes = np.arange(len(episode_rewards))
        ax1.plot(episodes, episode_rewards, linestyle='--', alpha=0.3,
                 color=color, label=f"{param_name}={val} (raw)")
        smoothed = np.convolve(episode_rewards, np.ones(10) / 10, mode='valid')
        ax1.plot(episodes[:len(smoothed)], smoothed, linestyle='-',
                 color=color, label=f"{param_name}={val} (smoothed)")

        alpha_color = color_cycle[(i + len(param_values)) % len(color_cycle)]
        ax2.plot(episodes, episode_alphas, linestyle=':', color=alpha_color,
                 label=f"{param_name} alpha")

    ax1.set_title(f"SAC: сравнение параметра {param_name} в FlappyBird-v0")
    ax1.set_xlabel("Эпизод")
    ax1.set_ylabel("Награда")
    ax2.set_ylabel("Alpha")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True)
    fig.tight_layout()
    plt.savefig(f"{filename_prefix}_FlappyBird-v0.png")
    plt.close()


def different_alpha():
    alpha = [0.1, 0.4, 0.8]
    run_and_plot(alpha, "alpha", lambda a: {"alpha": a}, "alpha")


def calibrate_alpha():
    is_auto_alpha = [True]
    run_and_plot(is_auto_alpha, "is_auto_alpha", lambda flag: {"is_auto_alpha": flag}, "is_auto_alpha")


def main():
    different_alpha()
    calibrate_alpha()


if __name__ == "__main__":
    main()
