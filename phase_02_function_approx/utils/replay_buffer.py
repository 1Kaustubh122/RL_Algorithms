import torch
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done, next_action=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done, next_action)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_actions = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        return states, actions, rewards, next_states, dones, next_actions

    def __len__(self):
        return len(self.buffer)



class NStepTransitionBuffer:
    def __init__(self, n, gamma, main_buffer):
        self.n = n
        self.gamma = gamma
        self.main_buffer = main_buffer
        self.bufer = deque()

    def push(self, state, action, reward, next_state, done):
        self.bufer.append((state, action, reward, next_state, done))
        
        if len(self.bufer) >=self.n:
            self._flush()

        if done:
            while self.bufer:
                self._flush(final=True)
                
    def _flush(self, final=False):
        G = 0
        for i, (_, _, reward, _, _) in enumerate(self.bufer):
            G += (self.gamma ** i) * reward
            
        state_0, action_0, _, _, _ = self.bufer[0]
        _, _, _, next_state_n, done_flag = self.bufer[-1]
        
        self.main_buffer.push(state_0, action_0, G, next_state_n, done_flag)
        
        self.bufer.popleft()
        
        if not final:
            return
