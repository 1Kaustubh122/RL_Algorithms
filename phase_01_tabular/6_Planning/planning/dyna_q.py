import numpy as np
from collections import defaultdict
from utils.policy import PolicySelection

class DynaQ:
    def __init__(self, env, policy, alpha=0.1, gamma=0.9, planning_steps=5):
        self.env = env
        self.alpha = alpha  
        self.gamma = gamma 
        self.planning_steps = planning_steps  
        self.num_action = len(self.env.action_space)
        self.Q = defaultdict(lambda: np.zeros(self.num_action))  
        self.model = defaultdict(lambda: defaultdict(lambda: (0.0, None)))  # (reward, next_state)
        self.seen_state_actions = []  
        self.policy = policy
        self.policy_selector = PolicySelection(
            action_space=self.env.action_space,
            policy_type=policy,
            epsilon=0.9
        )

    def action_selector(self, state, policy_type, Q_value=None):
  
        return self.policy_selector.select_action(state, policy_type, Q_value)
      
    def dyna_q(self, num_episodes=1000):
        for _ in range(num_episodes):
            state = self.env.reset()
            while True:
                action = self.action_selector(state, self.policy, self.Q)
                next_state, reward, done = self.env.step(action)
                max_q_next = 0.0 if done else np.max(self.Q[next_state])
                self.Q[state][action] += self.alpha * (
                    reward + self.gamma * max_q_next - self.Q[state][action]
                )
                self.model[state][action] = (reward, next_state)
                
                if (state, action) not in self.seen_state_actions:
                    self.seen_state_actions.append((state, action))
                    
                for _ in range(self.planning_steps):
                    
                    if not self.seen_state_actions:  
                        continue
                    
                    idx = np.random.randint(len(self.seen_state_actions))
                    s, a = self.seen_state_actions[idx]
                    r, s_next = self.model[s][a]
                    max_q_next_sim = 0.0 if s_next is None else np.max(self.Q[s_next])
                    self.Q[s][a] += self.alpha * (
                        r + self.gamma * max_q_next_sim - self.Q[s][a]
                    )
                    
                state = next_state
                if done:
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