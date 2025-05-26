import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.env import Enviornment  
from n_step.n_step_TD import n_step_td_pred

os.makedirs("Results", exist_ok=True)


def grid_gen(value_function, size=5):
    grid = np.full((size, size), np.nan)
    for state, value in value_function.items():
        r, c = state
        # if 0 <= r < size and 0 <= c < size:
        grid[r, c] = value
    return grid


def run_n_td_prediction(env, policy, num_episodes=1000, alpha=0.1, gamma=0.9, n=3):
    td_instance = n_step_td_pred(env=env, policy=policy, alpha=alpha, gamma=gamma, n=n)
    value_function = td_instance.td_pred(num_episode=num_episodes)
    grid = grid_gen(value_function, size=env.size)
    return grid


def plot_grid(ax, grid, title):
    """
    Visualize the value function grid with annotations.
    """
    image = ax.imshow(grid, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha='center', va='center', color='white')
    plt.colorbar(image, ax=ax, label='Value')

if __name__ == "__main__":
    size = 5
    env = Enviornment(size=size)

    val_grids = {}

    val_grids["n=2(Random Policy)"] = run_n_td_prediction(
        env, policy="random", num_episodes=1000, alpha=0.1, gamma=0.9, n=2
    )
    val_grids["n=3 (Random Policy)"] = run_n_td_prediction(
        env, policy="random", num_episodes=1000, alpha=0.1, gamma=0.9, n=3
    )

    num_plots = len(val_grids)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even for a single plot
    for ax, (method, grid) in zip(axes, val_grids.items()):
        plot_grid(ax, grid, method)
        
    fig.suptitle("Value Function n_step_td")
    fig.tight_layout()
    plt.savefig("Results/n_step_td.png")
    plt.show()
    
