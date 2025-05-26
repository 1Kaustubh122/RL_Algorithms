import os
import sys
import numpy as np
from collections import defaultdict
from utils.policy import PolicySelection

class td_pred:
    def __init__(self, env, policy, alpha=0.1, gamma=0.9):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.V = defaultdict(float)
        self.policy=policy
        self.policy_selector = PolicySelection(
            action_space=self.env.action_space,
            policy_type=policy,
            epsilon=0.9
        )

    
    def action_selector(self, state, policy_type, Q_value=None):
        return self.policy_selector.select_action(state, policy_type, Q_value)

        
    def gen_episode(self, policy_type, Q_value=None):
        episode = []
        state = self.env.reset()
        
        while True:
            action = self.action_selector(state, policy_type, Q_value)
            next_state, reward, done = self.env.step(self.env.action_space.index(action))

            episode.append((state, next_state, action, reward))
            state = next_state
            
            if done:
                break
        
        return episode
    
    def td_pred(self, num_episode=1000):
        for _ in range(num_episode):
            episode = self.gen_episode(policy_type=self.policy)
            for t in range(len(episode)):
                state, next_state, _, reward = episode[t]
                self.V[state] += self.alpha*(reward + (self.gamma*self.V[next_state])- self.V[state])
                
        return self.V
    
    def print_value_function(self):
        """Prints the value function as a grid."""
        for row in range(self.env.size):
            for col in range(self.env.size):
                state = (row, col)
                print(f"{self.V[state]:.2f}", end=" ")
            print()