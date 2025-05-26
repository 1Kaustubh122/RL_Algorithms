# **Handles N-Step SARSA (On & Off-Policy)**
import numpy as np
from collections import defaultdict
from utils.policy import PolicySelection

class N_STEP_SARSA:
    def __init__(self, env, policy, alpha=0.1, gamma=0.9, n=1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.num_action = len(self.env.action_space)
        self.Q = defaultdict(lambda: np.zeros(self.num_action))
        self.n = n
        self.policy=policy
        self.policy_selector = PolicySelection(
            action_space=self.env.action_space,
            policy_type=policy,
            epsilon=0.9
        )

    
    def action_selector(self, state, policy_type, Q_value=None):
        return self.policy_selector.select_action(state, policy_type, Q_value)

        
    def gen_episode(self, policy_type):
        episode = []
        state = self.env.reset()
        action_idx = self.action_selector(state, policy_type, self.Q)  
        while True:
            next_state, reward, done = self.env.step(action_idx)  
            if done:
                episode.append((state, next_state, action_idx, None, reward))
                break
            else:
                next_action_idx = self.action_selector(next_state, policy_type, self.Q)
                episode.append((state, next_state, action_idx, next_action_idx, reward))
                state = next_state  
                action_idx = next_action_idx  
        return episode
    
    def n_step_sarsa_on_policy(self, num_episode=1000):
        for _ in range(num_episode):
            episode = self.gen_episode(policy_type=self.policy)
            T = len(episode)  
            for t in range(T):  
                tau = t - self.n + 1  
                if tau >= 0:
                    state_tau, _, action_tau, _, _ = episode[tau]
                    G = 0.0
                    for i in range(tau + 1, min(tau + self.n, T)):
                        _, _, _, _, reward_i = episode[i]
                        G += (self.gamma ** (i - tau - 1)) * float(reward_i)
                    if tau + self.n < T:
                        _, state_tau_plus_n, _, action_tau_plus_n, _ = episode[tau + self.n]
                        if action_tau_plus_n is not None:
                            q_value = float(self.Q[state_tau_plus_n][action_tau_plus_n])
                            G += (self.gamma ** self.n) * q_value
                    self.Q[state_tau][action_tau] += self.alpha * (G - float(self.Q[state_tau][action_tau]))
                if tau >= T - 1:
                    break
        return self.Q
    
    def n_step_sarsa_off_policy(self, num_episode=1000):
        for _ in range(num_episode):
            episode = self.gen_episode(policy_type=self.policy)
            T = len(episode)  
            for t in range(T):  
                tau = t - self.n + 1  
                if tau >= 0:
                    state_tau, _, action_tau, _, _ = episode[tau]
                    G = 0.0
                    for i in range(tau + 1, min(tau + self.n, T)):
                        _, _, _, _, reward_i = episode[i]
                        G += (self.gamma ** (i - tau - 1)) * float(reward_i)
                    if tau + self.n < T:
                        _, state_tau_plus_n, _, action_tau_plus_n, _ = episode[tau + self.n]
                        if action_tau_plus_n is not None:
                            expected_value = 0.0
                            for a in range(self.num_action):
                                prob = self.policy_selector.get_policy_probab_e_greedy(
                                    state=state_tau_plus_n, action=a, policy_type=self.policy, Q=self.Q
                                )
                                expected_value += prob * self.Q[state_tau_plus_n][a]
                            G += (self.gamma ** self.n) * expected_value
                    self.Q[state_tau][action_tau] += self.alpha * (G - float(self.Q[state_tau][action_tau]))
                if tau >= T - 1:
                    break
        return self.Q
    
    def print_q_policy(self):
        grid_size = self.env.size
        action_map = {0: '↑', 1: '↓', 2: '→', 3: '←'} 
        for row in range(grid_size):
            row_str = []
            for col in range(grid_size):
                state = (row, col)
                best_action_idx = np.argmax(self.Q[state])
                row_str.append(action_map[best_action_idx])
            print(" | ".join(row_str))
        print("-" * (4 * grid_size))