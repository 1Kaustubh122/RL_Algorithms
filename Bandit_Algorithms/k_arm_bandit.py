import numpy as np
import matplotlib.pyplot as plt

'''
This is a simple implementation of K-Armed Bandit Problem

Greedy Estimate vs Upper-Confidence-Bound (UCB)

These are two types of action selection in RL

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
        # alpha = 1/self.action_counts[action]
        # self.q_estimates[action] += alpha * (reward - self.q_estimates[action])
        
        if alpha is None:  # Sample averaging
            alpha = 1/self.action_counts[action]
        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])
        
        

num_runs = 2000
num_steps = 1000
k=10

def loop(method, epsilion=0.1, c=2):
    avg_reward = np.zeros(num_steps)
    for _ in range(2000):
        bandit = Bandit(k)
        rewards = np.zeros(num_steps)
        
        for step in range(num_steps):
            
            if method == "epsilon_greedy":
                action = bandit.decaying_epsilon_greedy_sel(epsilion, step)     ## Uncomment this to test decayinf E-greedy
            elif method == "decaying_epsilon":
                action = bandit.epsilon_greedy(epsilion)
            elif method == "ucb":  
                action = bandit.ucb_action_selection(step, c)
            else:
                raise  ValueError("Method not found")
            reward = bandit.reward_function(action)
            bandit.update_action_value(action, reward, alpha=0.1)
            rewards[step] = reward
            
        avg_reward += rewards 
    
    return avg_reward / num_runs

methods = ["epsilon_greedy", "ucb"]
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

plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Performance of Epsilon-Greedy vs UCB")
plt.legend()
plt.grid()
plt.savefig("Bandit_Algorithms/Results/k_armed_bandit.png")
plt.show()