import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.env import Enviornment
from n_step.n_step_control import N_STEP_SARSA  
os.makedirs("TD/Results", exist_ok=True)

def grid_from_q(Q, size=5):
 
    grid = np.zeros((size, size))  # Init with zeros
    for state, q_values in Q.items():
        r, c = state
        grid[r, c] = np.max(q_values)  
    return grid

def run_n_step_sarsa(env, policy, num_episodes=1000, alpha=0.1, gamma=0.9, n=1):

    n_step_sarsa_instance = N_STEP_SARSA(env=env, policy=policy, alpha=alpha, gamma=gamma, n=n)
    Q = n_step_sarsa_instance.n_step_sarsa_on_policy(num_episode=num_episodes)
    grid = grid_from_q(Q, size=env.size)
    return grid

# Function to run n-step Expected SARSA and return the state-value grid
def run_n_step_expected_sarsa(env, policy, num_episodes=1000, alpha=0.1, gamma=0.9, n=1):
 
    n_step_sarsa_instance = N_STEP_SARSA(env=env, policy=policy, alpha=alpha, gamma=gamma, n=n)
    Q = n_step_sarsa_instance.n_step_sarsa_off_policy(num_episode=num_episodes)
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

    val_grids["n-Step SARSA (n=3)"] = run_n_step_sarsa(
        env, policy="e-greedy", num_episodes=5000, alpha=0.1, gamma=0.9, n=3
    )
    val_grids["n-Step Expected SARSA (n=3)"] = run_n_step_expected_sarsa(
        env, policy="e-greedy", num_episodes=5000, alpha=0.1, gamma=0.9, n=3
    )

    # Plotting
    num_plots = len(val_grids)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable for a single plot
    for ax, (method, grid) in zip(axes, val_grids.items()):
        plot_grid(ax, grid, method)
    fig.suptitle("n-Step SARSA vs off policy n-Step SARSA (ε-Greedy Policy)")
    fig.tight_layout()
    plt.savefig("Results/n_Step_SARSA_vs_ExpectedSARSA.png")
    plt.show()