# import os
# import matplotlib.pyplot as plt
# import sys
# import numpy as np

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from utils.env import Enviornment
# from utils.policy import PolicySelection
# from MC_Predection_and_Control.mc_predection import MonteCarloPrediction

# """
# Implementation of First-Visit Monte Carlo
# This method estimates value function using sample returns from the full episodes.
# It considers only first time a state 's' is visitited in an episode for updating the value function

# For each state, it computes G (sum of discounted rewards from that point onwards)
# It updates V(s) by taking average of all first visit returns.

# Drawbacks: Ignore extra info when states are visited multiple times,
#            Requires full episodes.
           
# Every-Visit:
# This Method also estimates the value function using sample returns from full episodes.
# it updates V(s) for every visit to a state in an episode, not just the first.

# For each statem it computes G similarly but this time it averages all returns across visits

# Drawbacks: May overweight repeated visits, still requires full episodes.
# """


# os.makedirs("Monte_Carlo/Results", exist_ok=True)


# def grid_gen(value_function, grid, size=5):
#     for row in range(size):
#         for col in range(size):
#             state = (row, col)
#             grid[row, col] = value_function[state]
        
#     return grid
            
            
# size = 5
# title = "State-Value Function"
# if __name__ == "__main__":

#     env = Enviornment(size)
# PolicySelection(action_space=env.action_space, epsilon=0.9)
#     mc_frist = MonteCarloPrediction(env, gamma=0.9)

#     ## For First Visit
#     V_first = mc_frist.first_visit_mc_on_policy(num_episodes=1000)
#     mc_frist.print_value_function()
#     grid_first = np.zeros((size, size))
    
#     grid_first = grid_gen(V_first, grid_first, size)
    
#     print()
    
#     ## For Every Visit
#     mc_every = MonteCarloPrediction(env, gamma=0.9)
    
#     V_every = mc_every.every_visit_mc_on_policy(num_episodes=100)
#     mc_every.print_value_function()
#     grid_every = np.zeros((size, size))
    
#     grid_every = grid_gen(V_every, grid_every, size)
    
    
#     ## For Off Policy MC
#     mc_off = MonteCarloPrediction(env, gamma=0.9, policy_type="e-greedy")
#     V_off = mc_off.off_policy_mc_pred(num_episodes=1000)
#     mc_off.print_value_function()
    
#     grid_off = np.zeros((size, size))
#     for row in range(size):
#         for col in range(size):
#             state = (row, col)
#             grid_off[row, col] = V_off[state]

            
#     # print(grid_every, grid_first)
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
#     image1 = axes[0].imshow(grid_first, cmap="viridis", interpolation = "nearest")
#     axes[0].set_title("First Visit e = 0.9")
#     axes[0].set_xlabel('Column')
#     axes[0].set_ylabel('Row')
#     axes[0].grid(True, which='both', color='black', linestyle='-', linewidth =0.5)
#     axes[0].set_xticks(np.arange(size))
#     axes[0].set_yticks(np.arange(size))
    
#     for i in range(size):
#         for j in range(size):
#             axes[0].text(j, i, f"{grid_first[i, j]:.2f}", ha='center', va='center', color='white')
    
#     fig.colorbar(image1, ax=axes[0], label='Value')
    
    
#     image2 = axes[1].imshow(grid_every, cmap="viridis", interpolation = "nearest")
#     axes[1].set_title("Every Visit e = 0.9")
#     axes[1].set_xlabel('Column')
#     # axes[1].set_ylabel('Row')
#     axes[1].grid(True, which='both', color='black', linestyle='-', linewidth =0.5)
#     axes[1].set_xticks(np.arange(size))
#     axes[1].set_yticks(np.arange(size))
    
#     for i in range(size):
#         for j in range(size):
#             axes[1].text(j, i, f"{grid_every[i, j]:.2f}", ha='center', va='center', color='white')
    
#     fig.colorbar(image2, ax=axes[2], label='Value')
    
    
#     image3 = axes[2].imshow(grid_every, cmap="viridis", interpolation = "nearest")
#     axes[2].set_title("Off Policy MC")
#     axes[2].set_xlabel('Column')
#     # axes[2].set_ylabel('Row')
#     axes[2].grid(True, which='both', color='black', linestyle='-', linewidth =0.5)
#     axes[2].set_xticks(np.arange(size))
#     axes[2].set_yticks(np.arange(size))
    
#     for i in range(size):
#         for j in range(size):
#             axes[2].text(j, i, f"{grid_every[i, j]:.2f}", ha='center', va='center', color='white')
    
#     fig.colorbar(image2, ax=axes[2], label='Value')
    
#     plt.tight_layout()
#     plt.savefig("Monte_Carlo/Results/First_vs_Every_Visit_MC.png") 
#     plt.show()




import os
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.env import Enviornment
from utils.policy import PolicySelection
from MC_Predection_and_Control.mc_predection import MonteCarloPrediction

os.makedirs("Monte_Carlo/Results", exist_ok=True)

def grid_gen(value_function, size=5):
    grid = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            state = (row, col)
            grid[row, col] = value_function[state]
    return grid

size = 5

env = Enviornment(size)
mc = MonteCarloPrediction(env, gamma=0.9)
mc1 = MonteCarloPrediction(env, gamma=0.9)
mc2 = MonteCarloPrediction(env, gamma=0.9)
mc3 = MonteCarloPrediction(env, gamma=0.9)

# First-Visit MC
V_first = mc.first_visit_mc_pred(num_episodes=1)
mc.print_value_function()
grid_first = grid_gen(V_first, size)
print()

# Every-Visit MC
V_every = mc1.every_visit_mc_pred(num_episodes=1)
mc1.print_value_function()
grid_every = grid_gen(V_every, size)
print()

# Off-Policy MC
mc2_off = MonteCarloPrediction(env, gamma=0.9, policy_type="e-greedy")
V_off = mc2_off.on_policy_mc_pred_Q_s_a(num_episodes=1)
mc2_off.print_q_policy()
print()

# Incremental MC
V_incremental = mc3.off_policy_mc_pred_V_pi(num_episodes=10000)
mc3.print_value_function()
grid_incremental = grid_gen(V_incremental, size)
print()

# # Weighted Importance Sampling MC
# V_weighted = mc.off_policy_mc_pred_Q_s_a(num_episodes=1000)
# mc.print_value_function()
# grid_weighted = grid_gen(V_weighted, size)

# # Visualization
# fig, axes = plt.subplots(1, 5, figsize=(30, 6))

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
#     fig.colorbar(image, ax=ax, label='Value')

# plot_grid(axes[0], grid_first, "First-Visit MC (e=0.9)")
# plot_grid(axes[1], grid_every, "Every-Visit MC (e=0.9)")
# plot_grid(axes[2], grid_off, "Off-Policy MC")
# plot_grid(axes[3], grid_incremental, "Incremental MC")
# plot_grid(axes[4], grid_weighted, "Weighted Importance Sampling MC")

# plt.tight_layout()
# plt.savefig("Monte_Carlo/Results/Monte_Carlo_Comparisons.png")
# plt.show()
