import os
import sys
import numpy as np
from collections import defaultdict

# Adjust the path to import PolicySelection from utils.policy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.policy import PolicySelection

# Ensure the results directory exists
os.makedirs("Monte_Carlo/Results", exist_ok=True)

class MonteCarloControl:
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_selector = PolicySelection(action_space=self.env.action_space, policy_type="e-greedy", epsilon=epsilon)
        self.num_action = len(self.env.action_space)             # Number of actions (4)
        self.Q = defaultdict(lambda: np.zeros(self.num_action))  # Q-table with integer indices
        self.C = defaultdict(lambda: np.zeros(self.num_action))  # Cumulative weights
        self.policy = defaultdict(int)                           # Policy maps states to integer action indices
        self.returns = defaultdict(list)                         # For storing returns

    def action_selector(self, state, policy_type, Q_Value=None):
        """Select an action and convert string to integer index."""
        action_str = self.policy_selector.select_action(state, policy_type, Q_Value)
        action_idx = self.env.action_space.index(action_str)  
        return action_idx

    def gen_episode(self, policy_type):
        """Generate an episode using the specified policy type."""
        episode = []
        state = self.env.reset()

        while True:
            action_idx = self.action_selector(state, policy_type, self.Q)  
            next_state, reward, done = self.env.step(action_idx)  
            episode.append((state, action_idx, reward)) 
            state = next_state
            if done:
                break

        return episode

    def mc_control_exploring_start(self, num_episodes=1000):
        """Monte Carlo control with exploring starts."""
        for _ in range(num_episodes):
            episode = self.gen_episode(policy_type="random")  # Random policy for exploration
            G = 0
            visited_state_action = set()

            for t in reversed(range(len(episode))):
                state, action_idx, reward = episode[t]
                G = self.gamma * G + reward

                if (state, action_idx) not in visited_state_action:
                    visited_state_action.add((state, action_idx))
                    self.returns[(state, action_idx)].append(G)
                    self.Q[state][action_idx] = np.mean(self.returns[(state, action_idx)])
                    self.policy[state] = np.argmax(self.Q[state])  # Update policy with best action

        return self.Q

    # def off_policy_mc_control(self, num_episodes=10000):
    #     """Off-policy Monte Carlo control using e-greedy as behavior policy."""
    #     for _ in range(num_episodes):
    #         episode = self.gen_episode(policy_type="e-greedy")  # Behavior policy
    #         G = 0.0
    #         W = 1.0

    #         for t in reversed(range(len(episode))):
    #             state, action_idx, reward = episode[t]
    #             G = self.gamma * G + reward

    #             self.C[state][action_idx] += W
    #             self.Q[state][action_idx] += (W / self.C[state][action_idx]) * (G - self.Q[state][action_idx])

    #             best_action_idx = np.argmax(self.Q[state])
    #             self.policy[state] = best_action_idx

    #             if action_idx != best_action_idx:
    #                 break

    #             b_a_s = self.policy_selector.get_policy_probab(state, action_idx, self.Q)
    #             if b_a_s == 0:
    #                 break

    #             W *= 1.0 / b_a_s

    #     return self.policy, self.Q

    def print_q_policy(self):
        """Print the policy as a grid with arrows."""
        grid_size = self.env.size
        action_map = {0: '↑', 1: '↓', 2: '→', 3: '←'}  # Map indices to arrows
        for row in range(grid_size):
            row_str = []
            for col in range(grid_size):
                state = (row, col)
                best_action_idx = np.argmax(self.Q[state])
                row_str.append(action_map[best_action_idx])
            print(" | ".join(row_str))
        print("-" * (4 * grid_size))