import heapq
import numpy as np
from collections import defaultdict
from utils.policy import PolicySelection

class PrioritizedSweeping:
    def __init__(self, env, policy, alpha=0.1, gamma=0.9, planning_steps=5, theta=0.01):
        self.env = env
        self.alpha = alpha 
        self.gamma = gamma
        self.planning_steps = planning_steps  
        self.theta = theta
        self.num_action = len(self.env.action_space)
        self.Q = defaultdict(lambda: np.zeros(self.num_action))  
        self.model = {} 
        self.predecessors = defaultdict(set) 
        self.pqueue = []
        self.pqueue_counter = 0  
        self.policy = policy
        self.policy_selector = PolicySelection(
            action_space=self.env.action_space,
            policy_type=policy,
            epsilon=0.1
        )

    def action_selector(self, state, policy_type, Q_value=None):
        return self.policy_selector.select_action(state, policy_type, Q_value)
        
    def prioritized_sweeping(self, num_episodes=1000):
        for _ in range(num_episodes):
            state = self.env.reset()
            while True:
                action = self.action_selector(state, self.policy, self.Q)
                next_state, reward, done = self.env.step(action)

                self.model[(state, action)] = (reward, next_state)
                if not done:
                    self.predecessors[next_state].add((state, action))

                max_q_next = 0.0 if done else np.max(self.Q[next_state])
                priority = abs(reward + self.gamma * max_q_next - self.Q[state][action])
                if priority > self.theta:
                    heapq.heappush(self.pqueue, (-priority, self.pqueue_counter, state, action))
                    self.pqueue_counter += 1

                for _ in range(self.planning_steps):
                    if not self.pqueue:
                        break
                    _, _, s, a = heapq.heappop(self.pqueue)
                    r, s_next = self.model[(s, a)]
                    max_q = 0.0 if s_next is None else np.max(self.Q[s_next])
                    self.Q[s][a] += self.alpha * (r + self.gamma * max_q - self.Q[s][a])
                    if s_next in self.predecessors:
                        for s_bar, a_bar in self.predecessors[s_next]:
                            r_bar, _ = self.model[(s_bar, a_bar)]
                            max_q_bar = 0.0 if s_next is None else np.max(self.Q[s_next])
                            priority = abs(r_bar + self.gamma * max_q_bar - self.Q[s_bar][a_bar])
                            if priority > self.theta:
                                heapq.heappush(self.pqueue, (-priority, self.pqueue_counter, s_bar, a_bar))
                                self.pqueue_counter += 1

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
