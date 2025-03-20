import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.env import Enviornment
from utils.policy import PolicySelection
from MC_Predection_and_Control.mc_predection import MonteCarloPrediction
import numpy as np

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

size = 5
title = "State-Value Function"
if __name__ == "__main__":

    env = Enviornment(size)
    policy_selector = PolicySelection(action_space=env.action_space, epsilon=0.9)
    mc = MonteCarloPrediction(policy_selector, env, gamma=0.9, policy_type="e-greedy")

    ## For First Visit
    V_first = mc.first_visit_mc_on_policy(num_episodes=1000)
    mc.print_value_function()
    grid_first = np.zeros((size, size))
    
    for row in range(size):
        for col in range(size):
            state = (row, col)
            grid_first[row, col] = V_first[state]
    print()
    
    V_every = mc.every_visit_mc_on_policy(num_episodes=1000)
    mc.print_value_function()
    
    grid_every = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            state = (row, col)
            grid_every[row, col] = V_every[state]
            
    V_off = mc.off_policy_mc_pred(num_episodes=1000)
    mc.print_value_function()
    
    grid_off = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            state = (row, col)
            grid_off[row, col] = V_off[state]

            
    # print(grid_every, grid_first)
    fig, (ax1, ax2) = plt.subplots(1,2 , figsize=(16, 6))
    
    image1 = ax1.imshow(grid_first, cmap="viridis", interpolation = "nearest")
    ax1.set_title("First Visit e = 0.9")
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.grid(True, which='both', color='black', linestyle='-', linewidth =0.5)
    ax1.set_xticks(np.arange(size))
    ax1.set_yticks(np.arange(size))
    
    for i in range(size):
        for j in range(size):
            ax1.text(j, i, f"{grid_first[i, j]:.2f}", ha='center', va='center', color='white')
    
    fig.colorbar(image1, ax=ax1, label='Value')
    
    
    image2 = ax2.imshow(grid_every, cmap="viridis", interpolation = "nearest")
    ax2.set_title("Every Visit e = 0.9")
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.grid(True, which='both', color='black', linestyle='-', linewidth =0.5)
    ax2.set_xticks(np.arange(size))
    ax2.set_yticks(np.arange(size))
    
    for i in range(size):
        for j in range(size):
            ax2.text(j, i, f"{grid_every[i, j]:.2f}", ha='center', va='center', color='white')
    
    fig.colorbar(image2, ax=ax2, label='Value')
    
    image3 = ax2.imshow(grid_every, cmap="viridis", interpolation = "nearest")
    ax2.set_title("Every Visit e = 0.9")
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.grid(True, which='both', color='black', linestyle='-', linewidth =0.5)
    ax2.set_xticks(np.arange(size))
    ax2.set_yticks(np.arange(size))
    
    for i in range(size):
        for j in range(size):
            ax2.text(j, i, f"{grid_every[i, j]:.2f}", ha='center', va='center', color='white')
    
    fig.colorbar(image2, ax=ax2, label='Value')
    
    plt.tight_layout()
    
    plt.savefig("Monte_Carlo/Results/First_vs_Every_Visit_MC.png") 

    plt.show()
