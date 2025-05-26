import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.env import Enviornment
from planning.dyna_q import DynaQ  
os.makedirs("Results", exist_ok=True)

def grid_from_q(Q, size=5):
    grid = np.zeros((size, size))
    for state, q_values in Q.items():
        r, c = state
        grid[r, c] = np.max(q_values)  
    return grid

def run_dyna_q(env, policy, num_episodes=1000, alpha=0.1, gamma=0.9, planning_steps=5):
   
    dyna_q_instance = DynaQ(env=env, policy=policy, alpha=alpha, gamma=gamma, planning_steps=planning_steps)
    Q = dyna_q_instance.dyna_q(num_episodes=num_episodes)
    grid = grid_from_q(Q, size=env.size)
    return grid

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

    val_grids["Dyna-Q"] = run_dyna_q(
        env, policy="e-greedy", num_episodes=5000, alpha=0.1, gamma=0.9, planning_steps=0
    )

    num_plots = len(val_grids)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]  
    for ax, (method, grid) in zip(axes, val_grids.items()):
        plot_grid(ax, grid, method)
    fig.suptitle("Dyna-Q State-Value Functions (ε-Greedy Policy)")
    fig.tight_layout()
    plt.savefig("Results/DynaQ.png")
    plt.show()