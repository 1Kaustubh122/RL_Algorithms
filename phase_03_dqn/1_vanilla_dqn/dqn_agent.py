import torch
import random
import numpy as np

DISCRETE_ACTIONS = [
    np.array([-1.0]),
    np.array([-0.5]),
    np.array([-0.25]),
    np.array([ 0.0]),
    np.array([ 0.25]),
    np.array([ 0.5]),
    np.array([ 1.0])
]
EPISODE_MAX_LEN = 300

def discrete_action(idx):
    return DISCRETE_ACTIONS[idx] 

def pick_random_action():
    return DISCRETE_ACTIONS[random.randint(0,  6)]

class DQNAgent:
    def __init__(self, q_net, target_net, replay_buffer, optimizer, loss_fn, epsilon, gamma, target_update_freq):
        self.q_net = q_net
        self.target_net = target_net
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.loss_fn = loss_fn 
        self.epsilon = epsilon
        self.gamma = gamma 
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0
        
    def e_greedy_action_select(self, state_tensor):
        if random.random() < self.epsilon:
            action = pick_random_action()
        else:
            # print(state_tensor.shape)
            q_value = self.q_net(state_tensor)
            
            argmax_q_value = torch.argmax(q_value).item()
            
            action = discrete_action(argmax_q_value)
        
        return action, DISCRETE_ACTIONS.index(action)
    
    def store_transition(self, state, action_idx, reward, next_state, done):
        self.replay_buffer.push(state, action_idx, reward, next_state, done, None)
        
    def learn(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        states, action_idxs, rewards, next_states, dones, _ = self.replay_buffer.sample(batch_size)
        
        if states.dim() == 3:
            states = states.squeeze(1)
        if next_states.dim() == 3:
            next_states = next_states.squeeze(1)
        
        if not isinstance(action_idxs, torch.Tensor):
            action_idxs = torch.as_tensor(action_idxs, dtype=torch.long, device=self.device)
        elif action_idxs.dtype != torch.long:
            action_idxs = action_idxs.long()
        
        q_values = self.q_net(states)
        q_values_selected = q_values.gather(1, action_idxs.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_value = self.target_net(next_states)  
            max_next_q_value, _ = next_q_value.max(dim=1) 
            target_q_values = (rewards + self.gamma * max_next_q_value * (1 - dones.float())).detach()

        
        loss = self.loss_fn(q_values_selected, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())