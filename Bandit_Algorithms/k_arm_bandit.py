import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("Bandit_Algorithms/Results", exist_ok=True)


'''
This is a simple implementation of K-Armed Bandit Problem

Upper-Confidence-Bound (UCB) vs Decaying Greedy Epsilon vs Epsilon-Greedy Selection

These are two types of action selection in RL

Drawbacks of UCB: Its smart but, assume the rewards for each arm wont chang over time.
For Non stationary problem, Thompson Sampling or decaying epsilon-greedy might work better

q_star = True Action Value
'''
class Bandit:
    def __init__(self, k):
        self.k = k
        self.q_estimates = np.ones(k) 
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
    
    def ucb_action_selection(self, step, c):
        return np.argmax(self.q_estimates + c * np.sqrt(np.log(step + 1)/(self.action_counts + 1)))
        
    def update_action_value(self, action, reward, alpha=None):
        self.action_counts[action] += 1
        if alpha is None:  # Sample averaging
            alpha = 1/self.action_counts[action]
        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])
        
        

num_runs = 2000
num_steps = 1000
k=10

def loop(method, epsilon=0.1, c=2):   ## Tune value of epsilon to see changes
    avg_reward = np.zeros(num_steps)
    for _ in range(num_runs):
        bandit = Bandit(k)
        rewards = np.zeros(num_steps)
        
        for step in range(num_steps):
            
            if method == "ucb":
                action = bandit.ucb_action_selection(step, c)
            elif method == "decay_epsilon_greedy":  
                action = bandit.decaying_epsilon_greedy_sel(epsilon, step)
            elif method == "epsilon_greedy":  
                action = bandit.epsilon_greedy(epsilon)
            else:
                raise  ValueError("Method not found")
            reward = bandit.reward_function(action)
            bandit.update_action_value(action, reward, alpha=0.1)
            rewards[step] = reward
            
        avg_reward += rewards 
    
    return avg_reward / num_runs

methods = ["epsilon_greedy", "ucb", "decay_epsilon_greedy"]
results_dict = {}
for method in methods:
    results_dict[method] = loop(method)

# Save results
np.save("Bandit_Algorithms/Results/k_armed_bandit.npy", results_dict)
load_res = np.load("Bandit_Algorithms/Results/k_armed_bandit.npy", allow_pickle=True).item()

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(load_res["epsilon_greedy"], label="Epsilon-Greedy (0.1)", color="blue")
plt.plot(load_res["ucb"], label="UCB (c=2)", color="red")
plt.plot(load_res["decay_epsilon_greedy"], label="Decaying Greedy Epsilon (0.1)", color="yellow")


plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Performance of Epsilon-Greedy vs UCB vs Decaying Greedy Epsilon")
plt.legend()
plt.grid()
plt.savefig("Bandit_Algorithms/Results/k_armed_bandit.png")
plt.show()