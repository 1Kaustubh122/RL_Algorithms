import os
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.env import Enviornment
from utils.policy import PolicySelection
from MC_Predection_and_Control.mc_predection import MonteCarloPrediction

"""
Implementation of First-Visit Monte Carlo
This method estimates value function using sample returns from the full episodes.
It considers only first time a state 's' is visitited in an episode for updating the value function

For each state, it computes G (sum of discounted rewards from that point onwards)
It updates V(s) by taking average of all first visit returns.

Drawbacks: Ignore extra info when states are visited multiple times,
           Requires full episodes.
           
Every-Visit:
This Method also estimates the value function using sample returns from full episodes.
it updates V(s) for every visit to a state in an episode, not just the first.

For each statem it computes G similarly but this time it averages all returns across visits

Drawbacks: May overweight repeated visits, still requires full episodes.
"""


os.makedirs("Monte_Carlo/Results", exist_ok=True)


def grid_gen(value_function, grid, size=5):
    for row in range(size):
        for col in range(size):
            state = (row, col)
            grid[row, col] = value_function[state]
        
    return grid
            
            
size = 5
title = "State-Value Function"
if __name__ == "__main__":

    env = Enviornment(size)
    policy_selector = PolicySelection(action_space=env.action_space, epsilon=0.9)
    mc_frist = MonteCarloPrediction(policy_selector, env, gamma=0.9)

    ## For First Visit
    V_first = mc_frist.first_visit_mc_on_policy(num_episodes=1000)
    mc_frist.print_value_function()
    grid_first = np.zeros((size, size))
    
    grid_first = grid_gen(V_first, grid_first, size)
    
    print()
    
    ## For Every Visit
    mc_every = MonteCarloPrediction(policy_selector, env, gamma=0.9)
    
    V_every = mc_every.every_visit_mc_on_policy(num_episodes=100)
    mc_every.print_value_function()
    grid_every = np.zeros((size, size))
    
    grid_every = grid_gen(V_every, grid_every, size)
    
    
    ## For Off Policy MC
    mc_off = MonteCarloPrediction(policy_selector, env, gamma=0.9, policy_type="e-greedy")
    V_off = mc_off.off_policy_mc_pred(num_episodes=1000)
    mc_off.print_value_function()
    
    grid_off = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            state = (row, col)
            grid_off[row, col] = V_off[state]

            
    # print(grid_every, grid_first)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    image1 = axes[0].imshow(grid_first, cmap="viridis", interpolation = "nearest")
    axes[0].set_title("First Visit e = 0.9")
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    axes[0].grid(True, which='both', color='black', linestyle='-', linewidth =0.5)
    axes[0].set_xticks(np.arange(size))
    axes[0].set_yticks(np.arange(size))
    
    for i in range(size):
        for j in range(size):
            axes[0].text(j, i, f"{grid_first[i, j]:.2f}", ha='center', va='center', color='white')
    
    fig.colorbar(image1, ax=axes[0], label='Value')
    
    
    image2 = axes[1].imshow(grid_every, cmap="viridis", interpolation = "nearest")
    axes[1].set_title("Every Visit e = 0.9")
    axes[1].set_xlabel('Column')
    # axes[1].set_ylabel('Row')
    axes[1].grid(True, which='both', color='black', linestyle='-', linewidth =0.5)
    axes[1].set_xticks(np.arange(size))
    axes[1].set_yticks(np.arange(size))
    
    for i in range(size):
        for j in range(size):
            axes[1].text(j, i, f"{grid_every[i, j]:.2f}", ha='center', va='center', color='white')
    
    fig.colorbar(image2, ax=axes[2], label='Value')
    
    
    image3 = axes[2].imshow(grid_every, cmap="viridis", interpolation = "nearest")
    axes[2].set_title("Off Policy MC")
    axes[2].set_xlabel('Column')
    # axes[2].set_ylabel('Row')
    axes[2].grid(True, which='both', color='black', linestyle='-', linewidth =0.5)
    axes[2].set_xticks(np.arange(size))
    axes[2].set_yticks(np.arange(size))
    
    for i in range(size):
        for j in range(size):
            axes[2].text(j, i, f"{grid_every[i, j]:.2f}", ha='center', va='center', color='white')
    
    fig.colorbar(image2, ax=axes[2], label='Value')
    
    plt.tight_layout()
    plt.savefig("Monte_Carlo/Results/First_vs_Every_Visit_MC.png") 
    plt.show()
