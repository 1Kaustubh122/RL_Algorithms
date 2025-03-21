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
            episode = gen_episode()
    
    def first_visit_mc_on_policy(self, num_episodes = 1000):
        for _ in range(num_episodes):
            episode = self.episode_generation(policy_type="random")
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
    
    def every_visit_mc_on_policy(self, num_episodes =1000):
        for _ in range(num_episodes):
            episode = self.episode_generation(policy_type="random")
            G = 0
            
            for t in reversed(range(len(episode))):
                state, _, reward = episode[t]
                G = self.gamma*G + reward

                self.returns[state].append(G)
                self.V[state] = np.mean(self.returns[state])
        
        return self.V
    
    def first_visit_off_policy(self, num_episode = 1000):
        C = defaultdict(float)
        
        for _ in range(num_episode):
            episode = self.episode_generation(policy_type="off_policy_behavior")
            G = 0
            W = 1
            visited_= set()
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma*G + reward
                
                if state not in visited_:
                    visited_.add(state)
                    self.sum_weights[(state, action)] += W
                    self.Q[(state, action)] += W/self.C[(state, action)] * (G - self.Q[(state, action)])
                    W *= self.policy_selector.target_policy(self.Q) / self.policy_selector.behavior_policy(self.Q)
                
                if W == 0:
                    break
            
            return self.Q
        
    def every_visit_off_policy(self, num_episode = 1000):
        C = defaultdict(float)
        
        for _ in range(num_episode):
            episode = self.episode_generation(policy_type="off_policy_behavior")
            G = 0
            W = 1
            
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma*G + reward
                self.sum_weights[(state, action)] += W
                self.Q[(state, action)] += W/self.C[(state, action)] * (G - self.Q[(state, action)])
                W *= self.policy_selector.target_policy(self.Q) / self.policy_selector.behavior_policy(self.Q)
                
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

