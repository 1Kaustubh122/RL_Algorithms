import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.env import Enviornment  # Ensure spelling matches your project files
from MC_Predection_and_Control.mc_predection import MonteCarloPrediction

os.makedirs("Monte_Carlo/Results", exist_ok=True)

def grid_gen(value_function, size=5):

    grid = np.full((size, size), np.nan)
    for state, value in value_function.items():
        r, c = state
        if 0 <= r < size and 0 <= c < size:
            grid[r, c] = value
    return grid

def grid_from_q(Q, size=5):
    grid = np.full((size, size), -np.inf)
    for (state, _), value in Q.items():
        r, c = state
        if 0 <= r < size and 0 <= c < size:
            grid[r, c] = max(grid[r, c], value)
    return grid


size = 5
gamma = 0.9
num_episodes = 1000
env = Enviornment(size)
m = "method"
pf = "print_func"
pvf = "print_value_function"

## q - Q_Policy
mc_methods = {
    "First-Visit MC": {m: "first_visit_mc_pred", pf: pvf, "type": "value"},
    "Every-Visit MC": {m: "every_visit_mc_pred", pf: pvf, "type": "value"},
    "On-Policy MC": {m: "on_policy_mc_pred_Q_s_a", pf: "print_q_policy", "type": "q"},
    "Off-Policy Vπ MC": {m: "off_policy_mc_pred_V_pi", pf: pvf, "type": "value"},
    "Weighted Importance Sampling MC": {m: "off_policy_mc_pred_Q_s_a", pf: "print_q_policy", "type": "q"},
}

val_grid = {}
q_grid = {}

for mc_type, config in mc_methods.items():
    print(f"\n===== Running : {mc_type} =====")
    
    mc_instance = MonteCarloPrediction(env, gamma=gamma)
    
    result = getattr(mc_instance, config[m])(num_episodes=num_episodes)
    getattr(mc_instance, config[pf])()
    
    if config["type"] == "value":
        grid = grid_gen(result, size)
        val_grid[mc_type] = grid
    elif config["type"] == "q":
        grid = grid_from_q(result, size)
        q_grid[mc_type] = grid



def plot_grid(ax, grid, title):
    image = ax.imshow(grid, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    for i in range(size):
        for j in range(size):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha='center', va='center', color='white')
    plt.colorbar(image, ax=ax, label='Value')


if val_grid:
    num_plots = len(val_grid)
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
    if num_plots == 1:
        axes = [axes]
    for ax, (method, grid) in zip(axes, val_grid.items()):
        plot_grid(ax, grid, method)
    fig.suptitle("Value Function Comparisons")
    fig.tight_layout()
    plt.savefig("Monte_Carlo/Results/Value_Functions_Comparison.png")
    plt.show()


if q_grid:
    num_plots = len(q_grid)
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
    if num_plots == 1:
        axes = [axes]
    for ax, (method, grid) in zip(axes, q_grid.items()):
        plot_grid(ax, grid, method)
    fig.suptitle("Q-Policy Derived Value Comparisons")
    fig.tight_layout()
    plt.savefig("Monte_Carlo/Results/Q_Policies_Comparison.png")
    plt.show()
