import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("MDP/Results", exist_ok=True)

"""
Implementation of Markov Decision Process.
Its is the foundation of RL, it consists of:
1) State (S)
2) Action (A)
3) Transition Probabilities (P)
4) Rewards (R)
5) Discount Factor (Y)

A transition function which returns the next state and the reward based on the current state and action.
This can be deterministic or probablistic.

Applying Bellman Equation to implement Value and Policy Iteration.
"""


class MDP:
    def __init__(self, discount_factor = 0.9):
        self.state_space = [(i, j) for i in range(3) for j in range(3)]
        self.actions = ["up", "down", "right", "left"]
        self.discount_factor = discount_factor
        self.rewards = {(0,2): 1, (2,2):-1}
        self.value_function = np.zeros((3,3))
        
        for state, reward in self.rewards.items():
            self.value_function[state[0], state[1]] = reward
        
    def transition_function(self, state, action):
        
        if state in self.rewards:
            return  state, 0
    
        row, col = state

        if action == "up":
            next_state = (max(row-1, 0), col)
        elif action == "down":
            next_state = (min(row+1, 2), col)
        elif action == "right":
            next_state = (row, min(col+1, 2))
        elif action == "left":
            next_state = (row, max(col-1, 0))
        else:
            next_state = state
            
        reward = self.rewards.get(next_state, 0)
        
        return next_state, reward
    
    def value_iteration_function(self, theta=10e-4):
        
        while True:
            max_change = 0
            for state in self.state_space:
                if state in self.rewards:
                    continue
                
                old_value = self.value_function[state[0], state[1]]
                action_value = []
                
                for action in self.actions:
                    next_state, reward = self.transition_function(state, action)
                    next_value = reward + self.discount_factor*self.value_function[next_state[0], next_state[1]]
                    action_value.append(next_value)
                
                new_value = max(action_value)
                self.value_function[state[0], state[1]] = new_value

                max_change = max(max_change, abs(old_value - new_value))

            if (max_change < theta):
                break
            
            
    def policy_extraction(self, state):
        best_action = None
        max_val = float('-inf')

        for action in self.actions:
            next_state, reward = self.transition_function(state, action)
            value = reward + self.discount_factor * self.value_function[next_state[0], next_state[1]]

            if value > max_val:
                max_val = value
                best_action = action
                
        return best_action
    


    def visualize_policy(self):
        action_arrows = {"up": "↑", "down": "↓", "right": "→", "left": "←"}
        policy_grid = np.full((3, 3), " ")

        for state in self.state_space:
            if state == (0,2):
                policy_grid[state] = "G"
            elif state == (2,2):
                policy_grid[state] = "X"
            else:
                best_action = self.policy_extraction(state)
                policy_grid[state] = action_arrows.get(best_action, " ")

        _, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks(np.arange(3) + 0.5, minor=True)
        ax.set_yticks(np.arange(3) + 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for i in range(3):
            for j in range(3):
                ax.text(j, i + 0.15, policy_grid[i, j], ha='center', va='center_baseline', 
                        fontsize=20, fontweight='bold')

        plt.gca().invert_yaxis()
        plt.subplots_adjust(bottom=0.1, top=0.9)  

        plt.draw() 
        plt.savefig("MDP/Results/mdp_3x3_grid.png") 
        plt.show()

        


mdp = MDP()
mdp.value_iteration_function()
mdp.visualize_policy()

