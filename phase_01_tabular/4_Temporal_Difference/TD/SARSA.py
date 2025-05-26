import numpy as np
from collections import defaultdict
from utils.policy import PolicySelection

class SARSA:
    def __init__(self, env, policy, alpha=0.1, gamma=0.9):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.num_action = len(self.env.action_space)
        self.Q = defaultdict(lambda: np.zeros(self.num_action))
        self.policy=policy
        self.policy_selector = PolicySelection(
            action_space=self.env.action_space,
            policy_type=policy,
            epsilon=0.9
        )

    
    def action_selector(self, state, policy_type, Q_value=None):
        return self.policy_selector.select_action(state, policy_type, Q_value)
        
    def gen_episode(self, policy_type):
        """Generate an episode using the specified policy type."""
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

    
    def sarsa(self, num_episode=1000):
        for _ in range(num_episode):
            episode = self.gen_episode(policy_type=self.policy)
            for t in range(len(episode)):
                state, next_state, action_idx, next_action_idx, reward = episode[t]
                if next_action_idx is None:
                    target = reward
                else:
                    target = reward + self.gamma * self.Q[next_state][next_action_idx]
                self.Q[state][action_idx] += self.alpha * (target - self.Q[state][action_idx])
        return self.Q
    
    def print_q_policy(self):
        """Print the policy as a grid with arrows."""
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