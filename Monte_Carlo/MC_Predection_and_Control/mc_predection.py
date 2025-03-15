import os
import numpy as np
from collections import defaultdict


os.makedirs("Monte_Carlo/Results", exist_ok=True)

class MonteCarloPrediction:
    def __init__(self, policy_selector, env, gamma = 0.9, policy_type="e-greedy"):
        self.policy_selector = policy_selector                # Policy Selector Instance
        self.env = env                                        # Environment
        self.gamma = gamma                                    # Discount Factor
        self.returns = defaultdict(list)                      # Stores returns after averaging
        self.V = defaultdict(float)                           # Stores State-Value Function
        self.policy_type = policy_type                        # Stores which policy to use
    
    def action_selector(self, Q_Value):
        """ Code to Select Policy Type"""
        if self.policy_type == "e-greedy":
            return self.policy_selector.e_greedy_policy(Q_Value)
        elif self.policy_type == "greedy":
            return self.policy_selector.greedy_sel(Q_Value)
        else:
            raise ValueError(" Choose 'e-greedy' or 'greedy' ")
        
      
            
    def episode_generation(self):
        episode = []
        state = self.env.reset()
        Q_values = defaultdict(lambda: np.zeros(len(self.env.action_space)))  
        
        while True:
            # self.env.render()
            action = self.action_selector(Q_values[state])                    

            next_state, reward, done = self.env.step(action)

            episode.append((state, action, reward))
            state = next_state
            # print("Done first ep")
            if done:
                break
            
        return episode
    
    def first_visit_mc_pred(self, num_episodes = 1000):
        for _ in range(num_episodes):
            episode = self.episode_generation()
            G = 0
            visited_states = set()
            
            for t in reversed(range(len(episode))):
                state, _, reward = episode[t]
                G = self.gamma*G + reward
                
                if state not in visited_states:
                    visited_states.add(state)
                    self.returns[state].append(G)
                    self.V[state] = np.mean(self.returns[state])
        
        return self.V
    
    def every_visit_mc_pred(self, num_episodes =1000):
        for _ in range(num_episodes):
            episode = self.episode_generation()
            G = 0
            
            for t in reversed(range(len(episode))):
                state, _, reward = episode[t]
                G = self.gamma*G + reward

                self.returns[state].append(G)
                self.V[state] = np.mean(self.returns[state])
        
        return self.V
    
    def print_value_function(self):
        """Prints the value function as a grid."""
        for row in range(self.env.size):
            for col in range(self.env.size):
                state = (row, col)
                print(f"{self.V[state]:.2f}", end=" ")
            print()

