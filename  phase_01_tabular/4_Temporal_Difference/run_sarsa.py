import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.env import Enviornment  
from TD.SARSA import SARSA

os.makedirs("Results", exist_ok=True)


def grid_from_q(Q, size=5):
    grid = np.full((size, size), -np.inf)
    grid = np.zeros((size, size))  # Initialize with zeros
    for state, q_values in Q.items():
        r, c = state
        grid[r, c] = np.max(q_values)  # V(s) = max_a Q(s, a)
    return grid


def run_sarsa(env, policy, num_episodes=1000, alpha=0.1, gamma=0.9):
    """
    Run TD prediction and return the value function as a grid.
    """
    sarsa_instance = SARSA(env=env, policy=policy, alpha=alpha, gamma=gamma)
    Q = sarsa_instance.sarsa(num_episode=num_episodes)
    grid = grid_from_q(Q, size=env.size)
    sarsa_instance.print_q_policy()
    return grid


def plot_grid(ax, grid, title):
    """
    Visualize the state-value grid with the annotations.
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

    val_grids["e-greedy policy"] = run_sarsa(
        env, policy="e-greedy", num_episodes=1000, alpha=0.1, gamma=0.9
    )

    num_plots = len(val_grids)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]  
    for ax, (method, grid) in zip(axes, val_grids.items()):
        plot_grid(ax, grid, method)
    fig.suptitle("SARSA State-Value Function")
    fig.tight_layout()
    plt.savefig("Results/SARSA_Value_Function.png")
    plt.show()
    
