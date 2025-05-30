import torch
import random
import numpy as np

DISCRETE_ACTIONS = [np.array([v]) for v in np.linspace(-1.0, 1.0, 9)]

print(DISCRETE_ACTIONS)
print(len(DISCRETE_ACTIONS))


def discrete_action(idx):
    return DISCRETE_ACTIONS[idx] 

def pick_random_action():
    return DISCRETE_ACTIONS[random.randint(0,  8)]

class PerDqnAgent:
    def __init__(self, q_net, target_net, replay_buffer, optimizer, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, gamma=0.99, target_update_freq=10):

        self.q_net = q_net
        self.target_net = target_net
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
    def e_greedy_action_select(self, state_tensor):
        if random.random() < self.epsilon:
            action = pick_random_action()
        else:
            # print(state_tensor.shape)
            q_value = self.q_net(state_tensor)
            
            argmax_q_value = torch.argmax(q_value).item()
            
            action = discrete_action(argmax_q_value)
        
        return action, DISCRETE_ACTIONS.index(action)
    

    def learn(self, batch_size = 64):
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(batch_size)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1-dones)
        
        td_error = target_q - q_values

        new_priorities = td_error.detach().abs().cpu().numpy()
        clipped = np.clip(new_priorities, 1e-4, 10.0)
        self.replay_buffer.update_priorities(indices, clipped)

        loss = (weights * td_error ** 2).mean()
        
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

        