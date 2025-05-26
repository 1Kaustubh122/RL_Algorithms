import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.env import Enviornment  
from MC_Predection_and_Control.mc_control import MonteCarloControl

os.makedirs("Monte_Carlo/Results", exist_ok=True)


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
epsilon = 0.9
env = Enviornment(size)
m = "method"
pf = "print_func"

## q - Q_Policy
mc_methods = {
    "Exploring Start MC": {m: "mc_control_exploring_start", pf: "print_q_policy"},
    # "Off Policy Control MC": {m: "off_policy_mc_control", pf: "print_q_policy"}
}

q_grid = {}

for mc_type, config in mc_methods.items():
    print(f"\n===== Running : {mc_type} =====")
    
    mc_instance = MonteCarloControl(env, gamma=gamma, epsilon=epsilon)
    
    result = getattr(mc_instance, config[m])(num_episodes=num_episodes)
    getattr(mc_instance, config[pf])()
    

    # grid = grid_from_q(result, size)
    # q_grid[mc_type] = grid



# def plot_grid(ax, grid, title):
#     image = ax.imshow(grid, cmap="viridis", interpolation="nearest")
#     ax.set_title(title)
#     ax.set_xlabel('Column')
#     ax.set_ylabel('Row')
#     ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
#     ax.set_xticks(np.arange(size))
#     ax.set_yticks(np.arange(size))
#     for i in range(size):
#         for j in range(size):
#             ax.text(j, i, f"{grid[i, j]:.2f}", ha='center', va='center', color='white')
#     plt.colorbar(image, ax=ax, label='Value')


# if q_grid:
#     num_plots = len(q_grid)
#     fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
#     if num_plots == 1:
#         axes = [axes]
#     for ax, (method, grid) in zip(axes, q_grid.items()):
#         plot_grid(ax, grid, method)
#     fig.suptitle("Q-Policy Derived Value Comparisons")
#     fig.tight_layout()
#     plt.savefig("Monte_Carlo/Results/MC_Control.png")
#     plt.show()
