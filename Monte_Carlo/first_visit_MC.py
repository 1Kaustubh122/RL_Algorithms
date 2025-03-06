import numpy as np

class FirstVisitMcPredection:
    def __init__(self, pi=None, gamma=0.9):
        self.state_space = [(i, j) for i in range(3) for j in range(3)]
        self.rewards = {(0,2): 1, (2,2):-1}
        self.val_pi_s = {s: self.rewards.get(s, 0) for s in self.state_space}
        self.episode_list = []
        self.gamma = gamma
        self.num_visits = {s : 0 for s in self.state_space}
        self.pi = self.pi = {
            (0, 0): "right", (0, 1): "right", (0, 2): "right",
            (1, 0): "up",    (1, 1): "up",    (1, 2): "right",
            (2, 0): "up",    (2, 1): "right", (2, 2): "up"
        }

        

    def transition_function(self, state, action):
        
        if state in self.rewards:
            return  state, self.rewards.get(state)
    
        row, col = state

        if action == "up":
            next_state = (max(row-1, 0), col)
        elif action == "down":
            next_state = (min(row+1, 2), col)
        elif action == "right":
            next_state = (row, min(col+1, 2))
        elif action == "left":
            next_state = (row, max(col-1, 0))
        else:
            next_state = state
            
        reward = self.rewards.get(next_state, 0)
        
        return next_state, reward
        
    def episode_gen(self, start_state, max_steps=100):
        
        state = start_state
        self.episode_list = []
        
        for _ in range(max_steps):
           
            action = self.pi[state]
            next_state, reward = self.transition_function(state, action) 
            self.episode_list.append((state, action, reward))
            
            if next_state in self.rewards:
                break
            state = next_state
    
    def monte_carlo(self):
        epsiode_lst = list(reversed(self.episode_list))
        
        G = 0
        visited_states = set()
        
        for state, action, reward in epsiode_lst:
            G = self.gamma * G + reward
            
            if state not in visited_states and state not in self.rewards:
                self.num_visits[state] += 1
                self.val_pi_s[state] += (G - self.val_pi_s[state]) / self.num_visits[state]
                visited_states.add(state)
                
            
    def train(self, num_episodes=1000):
        for i in range(num_episodes):
            st_state = self.state_space[np.random.randint(len(self.state_space))]
            self.episode_gen(st_state)
            self.monte_carlo()

            if i % 100 == 0:
                print(f"Iteration {i}, Values:")
                for row in range(3):
                    print([round(self.val_pi_s[(row, col)], 2) for col in range(3)])
                print("\n")


alpha = FirstVisitMcPredection()
alpha.train()