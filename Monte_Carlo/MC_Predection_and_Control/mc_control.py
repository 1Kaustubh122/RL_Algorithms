import os
import numpy as np
from collections import defaultdict
from mc_predection import MonteCarloPrediction

os.makedirs("Monte_Carlo/Results", exist_ok=True)

class MonteCarloControl:
    def __init__(self, policy_selector, env, gamma = 0.9):
        
        self.policy_selector = policy_selector
        self.env = env
        self.gamma = gamma
        self.returns = defaultdict(list)
        self.Q_S_A = defaultdict(lambda: np.zeros(len(self.env.action_space)))
        self.A_star = defaultdict(float)
     
        
    def action_selector(self, Q_values, policy_type):
        """ Selection of policy type"""
        if policy_type == "on_policy":
            return self.policy_selector.e_greedy_policy(Q_values)
        elif policy_type == "off_policy_behavior":
            return self.policy_selector.behavior_policy(Q_values)
        elif policy_type == "off_policy_target":
            return self.policy_selector.target_policy(Q_values)
        
    def episode_gen(self):
        episode = []
        state = self.env.reset()
        
        while True:
            Q_values = [self.Q_S_A[(state, a)] for a in self.env.action_space]
            action = self.action_selector(Q_values, "on_policy")
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode
    
    def exploring_start_mc(self):
        ...
            
    def on_policy_first_visit(self, num_episode= 1000):
        for _ in range(num_episode):
            episode = self.episode_gen()
            G = 0
            
            visited_state_action = set()
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma*G + reward
                
                if (state, action) not in visited_state_action:
                    visited_state_action.add((state, action))
                    
                    self.returns[(state, action)].append(G)
                    self.Q_S_A[(state, action)] = np.mean(self.returns[(state, action)])

                    self.A_star[state] = np.argmax([self.Q_S_A[state, a] for a in self.env.action_space])

                    for a in self.env.action_space:
                        if a == self.A_star[state]:
                            self.policy_selector.epsilon_policy[a] = 1 - self.policy_selector.epsilon + (self.policy_selector.epsilon / len(self.env.action_space))
                        else:
                            self.policy_selector.epsilon_policy[a] = self.policy_selector.epsilon/len(self.env.action_space)
                    