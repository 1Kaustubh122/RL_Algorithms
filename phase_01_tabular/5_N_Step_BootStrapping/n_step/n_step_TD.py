# - **Handles self.N-Step TD for state-value prediction**

from collections import defaultdict
from utils.policy import PolicySelection

class n_step_td_pred:
    def __init__(self, env, policy, alpha=0.1, gamma=0.9, n=1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.V = defaultdict(float)
        self.n = n
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
            episode.append((state, reward, next_state))
            state = next_state
            
            if done:
                break
        
        return episode
    
    def td_pred(self, num_episode=1000):
        for _ in range(num_episode):
            
            episode = self.gen_episode(policy_type=self.policy)
            T = len(episode)

            for t in range(T):
                tau = t - self.n + 1
                if tau >= 0:
                    state_tau, _, _= episode[tau]
                    G = 0
                    for i in range(tau + 1, min((tau + self.n), T)):
                        _, reward_i, _ = episode[i]
                        G += (self.gamma ** (i - tau - 1)) * reward_i
                        
                    if tau + self.n < T:
                        _, _, state_tau_plus_n = episode[tau + self.n]
                        G += (self.gamma ** self.n) * self.V[state_tau_plus_n]
                        
                    self.V[state_tau] += self.alpha * (G - self.V[state_tau])
                
                if tau == T-1:
                    break
                
        return self.V
    
    def print_value_function(self):
        """Prints the value function as a grid."""
        for row in range(self.env.size):
            for col in range(self.env.size):
                state = (row, col)
                print(f"{self.V[state]:.2f}", end=" ")
            print()