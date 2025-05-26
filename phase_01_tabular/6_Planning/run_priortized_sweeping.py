import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Adjust the path to import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.env import Enviornment
from planning.prioritized_sweeping import PrioritizedSweeping

os.makedirs("Results", exist_ok=True)

def grid_from_q(Q, size=5):
    grid = np.zeros((size, size))
    for state, q_values in Q.items():
        r, c = state
        grid[r, c] = np.max(q_values)  
    return grid


def run_prioritized_sweeping(env, policy, num_episodes=10, alpha=0.1, gamma=0.9, planning_steps=5, theta=0.01):
    ps_instance = PrioritizedSweeping(env=env, policy=policy, alpha=alpha, gamma=gamma, planning_steps=planning_steps, theta=theta)
    Q = ps_instance.prioritized_sweeping(num_episodes=num_episodes)
    return grid_from_q(Q, size=env.size)

def plot_grid(ax, grid, title):
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
    plt.colorbar(image, ax=ax, label='State Value')

if __name__ == "__main__":
    size = 5
    env = Enviornment(size=size)

    val_grids = {}

    val_grids["Prioritized Sweeping (5 planning steps)"] = run_prioritized_sweeping(
        env, policy="e-greedy", num_episodes=5000, alpha=0.1, gamma=0.9, planning_steps=5, theta=0.01
    )

    # Plotting
    num_plots = len(val_grids)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes] 
    for ax, (method, grid) in zip(axes, val_grids.items()):
        plot_grid(ax, grid, method)
    fig.suptitle("Prioritized Sweeping (ε-Greedy Policy)")
    fig.tight_layout()
    plt.savefig("Results/Prioritized_Sweeping.png")
    plt.show()