import numpy as np
import matplotlib.pyplot as plt

'''
This is a simple implementation of K-Armed Bandit Problem
q_star = True Action Value

'''
class Bandit:
    def __init__(self, k):
        self.k = k
        self.q_estimates = np.zeros(k)
        self.action_counts = np.zeros(k)
        self.q_star = np.random.normal(0, 1, self.k)
        
    def reward_function(self, action):
        return np.random.normal(self.q_star[action], 1)  ## Gaussian Noise
    
    def greedy_sel(self):
        return np.argmax(self.q_estimates)  ## Exploitation
    
    def epsilon_greedy(self, epsilon):
        
        if np.random.rand() < epsilon:
            return np.random.randint(self.k)   ## Exploration
        return self.greedy_sel()
        
    def decaying_epsilon_greedy_sel(self, epsilon, step):
        epsilon = 1/(1+(0.001 * step))   ## Smooth decay
        if np.random.rand() < epsilon:
            return np.random.randint(self.k)  
        return self.greedy_sel() 
        
    def update_action_value(self, action, reward):
        self.action_counts[action] += 1
        alpha = 1/self.action_counts[action]
        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])

num_runs = 2000
num_steps = 1000
k=10

def loop(epsilion):
    avg_reward = np.zeros(num_steps)
    for _ in range(2000):
        bandit = Bandit(k)
        rewards = np.zeros(num_steps)
        
        for step in range(1000):
            # action = bandit.decaying_epsilon_greedy_sel(epsilion, step)     ## Uncomment this to test decayinf E-greedy
            action = bandit.epsilon_greedy(epsilion)
            reward = bandit.reward_function(action)
            bandit.update_action_value(action, reward)
            rewards[step] = reward
            
        avg_reward += rewards 
    
    return avg_reward / num_runs

## Epsilon Value to test with
epsilon_values = [0, 0.01, 0.1, 0.5]
results_dict = {}

for eps in epsilon_values:
    results_dict[f"epsilon_{eps}"] = loop(eps)

np.save("Bandit_Algorithms/Results/k_armed_bandit.npy", results_dict)
load_res = np.load("Bandit_Algorithms/Results/k_armed_bandit.npy", allow_pickle=True).item()

plt.figure(figsize=(10, 5))
colors = ["red", "blue", "yellow", "green"]


for i, eps in enumerate(epsilon_values):
    plt.plot(load_res[f"epsilon_{eps}"], label=f"Epsilon = {eps}", color=colors[i])

plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Epsilon-Greedy Performance Over Time")
# plt.title("Decaying Epsilon-Greedy Performance Over Time")
plt.legend()
plt.grid()
plt.savefig("Bandit_Algorithms/Results/k_armed_bandit.png")
plt.show()

