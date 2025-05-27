##  Replay Buffer   ##
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        
        self.capacity = capacity
        self.state_shape = state_shape
        self.states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)

        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.index  = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def getSample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[index],
            self.actions[index],
            self.rewards[index],
            self.next_states[index],
            self.dones[index]
        )
        
    def __len__(self):
        return self.size