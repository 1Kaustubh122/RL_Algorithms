import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.env import Enviornment  
from TD.TD_Prediction import td_pred

os.makedirs("Results", exist_ok=True)


def grid_gen(value_function, size=5):
    grid = np.full((size, size), np.nan)
    for state, value in value_function.items():
        r, c = state
        # if 0 <= r < size and 0 <= c < size:
        grid[r, c] = value
    return grid


# size = 5
# gamma = 0.9
# num_episodes = 1000
# env = Enviornment(size)



def run_td_prediction(env, policy, num_episodes=1000, alpha=0.1, gamma=0.9):
    """
    Run TD prediction and return the value function as a grid.
    """
    td_instance = td_pred(env=env, policy=policy, alpha=alpha, gamma=gamma)
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

# Main execution
if __name__ == "__main__":
    # Set up environment
    size = 5
    env = Enviornment(size=size)

    # Dictionary to store value function grids (for flexibility with multiple policies)
    val_grids = {}

    # Run TD prediction with random policy
    val_grids["Random Policy"] = run_td_prediction(
        env, policy="random", num_episodes=1000, alpha=0.1, gamma=0.9
    )

    # Plotting
    num_plots = len(val_grids)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even for a single plot
    for ax, (method, grid) in zip(axes, val_grids.items()):
        plot_grid(ax, grid, method)
    fig.suptitle("Value Function Comparisons")
    fig.tight_layout()
    plt.savefig("Results/Value_Functions_Comparison.png")
    plt.show()
    
