import os
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.env import Environment  # Fixed typo in class name
from utils.policy import PolicySelection
from MC_Predection_and_Control.mc_predection import MonteCarloPrediction

os.makedirs("Monte_Carlo/Results", exist_ok=True)

size = 5
title = "State-Value Function"

if __name__ == "__main__":
    env = Environment(size)
    policy_selector = PolicySelection(action_space=env.action_space, epsilon=0.9)
    mc = MonteCarloPrediction(policy_selector, env, gamma=0.9, policy_type="e-greedy")

    # First-Visit MC
    V_first = mc.first_visit_mc_on_policy(num_episodes=1000)
    mc.print_value_function()
    grid_first = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            grid_first[row, col] = V_first[(row, col)]
    
    # Every-Visit MC
    V_every = mc.every_visit_mc_on_policy(num_episodes=1000)
    mc.print_value_function()
    grid_every = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            grid_every[row, col] = V_every[(row, col)]
    
    # Off-Policy MC
    V_off = mc.off_policy_mc_pred(num_episodes=1000)
    mc.print_value_function()
    grid_off = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            grid_off[row, col] = V_off[(row, col)]
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # First-Visit Plot
    img1 = axes[0].imshow(grid_first, cmap="viridis", interpolation="nearest")
    axes[0].set_title("First Visit MC (e=0.9)")
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    for i in range(size):
        for j in range(size):
            axes[0].text(j, i, f"{grid_first[i, j]:.2f}", ha='center', va='center', color='white')
    fig.colorbar(img1, ax=axes[0], label='Value')
    
    # Every-Visit Plot
    img2 = axes[1].imshow(grid_every, cmap="viridis", interpolation="nearest")
    axes[1].set_title("Every Visit MC (e=0.9)")
    axes[1].set_xlabel('Column')
    for i in range(size):
        for j in range(size):
            axes[1].text(j, i, f"{grid_every[i, j]:.2f}", ha='center', va='center', color='white')
    fig.colorbar(img2, ax=axes[1], label='Value')
    
    # Off-Policy Plot
    img3 = axes[2].imshow(grid_off, cmap="viridis", interpolation="nearest")
    axes[2].set_title("Off-Policy MC")
    axes[2].set_xlabel('Column')
    for i in range(size):
        for j in range(size):
            axes[2].text(j, i, f"{grid_off[i, j]:.2f}", ha='center', va='center', color='white')
    fig.colorbar(img3, ax=axes[2], label='Value')
    
    plt.tight_layout()
    plt.savefig("Monte_Carlo/Results/MC_Prediction_Comparison.png")
    plt.show()
