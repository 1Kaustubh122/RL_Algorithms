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
        self.Q_S_A = defaultdict(lambda: np.zeros(len(self.env.action_space)))
        self.policy = defaultdict(lambda: np.random.randint(len(self.env.action_space)))
        self.returns = defaultdict(lambda: [[] for _ in range(len(self.env.action_space))])
     
        
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
    
    def mc_control_exploring_start(self, num_episodes = 1000):
        for _ in range(num_episodes):
            start = self.env.reset()            ## Can update it to choose random action too #Default (0, 0)
            action = np.random.randint()
            
            
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
                    