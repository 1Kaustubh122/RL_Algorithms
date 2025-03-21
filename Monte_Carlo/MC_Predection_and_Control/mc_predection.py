import os
import numpy as np
from collections import defaultdict


os.makedirs("Monte_Carlo/Results", exist_ok=True)

class MonteCarloPrediction:
    def __init__(self, policy_selector, env, gamma = 0.9):
        self.policy_selector = policy_selector                # Policy Selector Instance
        self.env = env                                        # Environment
        self.gamma = gamma                                    # Discount Factor
        self.returns = defaultdict(list)                      # Stores returns after averaging
        self.sum_weighted_returns = defaultdict(float)        # Stores sum of weighted returns
        self.sum_weights = defaultdict(float)                 # Stores sum of weight  
        self.V = defaultdict(float)                           # Stores State-Value Function
        self.num_action = len(self.env.action_space)          # Number of possible action
        self.Q = defaultdict(lambda: np.zeros(self.num_action))
        
        
    def action_selector(self, state, policy_type, Q_Value=None):
        """ Code to Select Policy Type"""
        return self.policy_selector.select_action(state, policy_type, Q_Value)
        
            
    def gen_episode(self, policy_type, Q=None):
        episode = []
        state = self.env.reset()
        # Q_values = defaultdict(lambda: np.zeros(len(self.env.action_space)))  
        
        while True:
            # self.env.render()
            action = self.action_selector(state, policy_type, Q)                    

            next_state, reward, done = self.env.step(action)

            episode.append((state, action, reward))
            state = next_state
            # print("Done first ep")
            if done:
                break
            
        return episode
    
    def first_visit_mc_pred(self, num_episodes = 1000):
        for _ in range(num_episodes):
            episode = self.gen_episode(policy_type="random")
            G = 0
            visited_state = set()
            for t in range(reversed(range(len(episode)))):
                state, _, reward = episode[t]
                G = self.gamma*G + reward
                
                if state not in visited_state:
                    visited_state.add(state)
                    self.returns[state].append(G)
                    self.V[state] = np.mean(self.returns[state])
        
        return self.V
                
    def every_visit_mc_pred(self, num_episodes = 1000):
        for _ in range(num_episodes):
            episode = self.gen_episode(policy_type="random")
            G = 0
            for t in range(reversed(range(len(episode)))):
                state, _, reward = episode[t]
                G = self.gamma*G + reward
                self.returns[state].append(G)
                self.V[state] = np.mean(self.returns[state])
        
        return self.V
    
    def on_policy_mc_pred_Q_s_a(self, num_episodes = 1000):
        returns = defaultdict(lambda: [[] for _ in range(self.num_action)])
        for _ in range(num_episodes):
            episode = self.gen_episode(policy_type="random")
            G = 0
            visited_state = set()
            for t in range(reversed(episode)):
                state, action, reward = episode[t]
                G = self.gamma*G + reward
                
                if state not in visited_state:
                    visited_state.add((state, action))
                    returns[state][action].append(G)
                    self.Q[state][action] = np.mean(returns[state][action])
        
        return self.Q
            
    def off_policy_mc_pred_V_pi(self, num_episodes = 1000):
        for _ in range(num_episodes):
            episode = self.gen_episode(policy_type="off_policy_behavior")
            G = 0
            W = 1
            visited_state = set()
            for t in reversed(range(len(episode))):
                state, _, reward = episode[t]
                G = self.gamma*G + reward
                
                if state not in visited_state:
                    visited_state.add(state)
                    self.sum_weighted_returns[state] += W * G
                    self.sum_weights[state] += W
                    
                    if self.sum_weights[state] != 0:
                        self.V[state] = self.sum_weighted_returns[state] / self.sum_weights[state]
                
                pi_a = 1
                b_a = 1/self.num_action
                
                if b_a == 0:
                    break
                
                W *= pi_a/b_a
                
                if W == 0:
                    break
            
        return self.V
        
    def off_policy_mc_pred_Q_s_a(self, num_episodes = 1000):
        for _ in range(num_episodes):
            episode = self.gen_episode(policy_type="off_policy_behavior")
            G = 0
            W = 1
            visited_state = set()
            
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                if (state, action) not in visited_state:
                    visited_state.add((state, action))
                    self.sum_weighted_returns[state][action] += W * G
                    self.sum_weights[state][action] += W
                    
                    if self.sum_weights[state][action] != 0:
                        self.Q[state][action] = self.sum_weighted_returns[state][action] / self.sum_weights[state][action]
                        
                pi_a = 1
                b_a = 1/self.num_action
                
                if b_a == 0:
                    break
                
                W *= pi_a/b_a
                
                if W == 0:
                    break
                
        return self.Q
        
    def print_value_function(self):
        """Prints the value function as a grid."""
        for row in range(self.env.size):
            for col in range(self.env.size):
                state = (row, col)
                print(f"{self.V[state]:.2f}", end=" ")
            print()

