import torch
import random
import numpy as np

# DISCRETE_ACTIONS = [np.array([v]) for v in np.linspace(-1.0, 1.0, 5)]
DISCRETE_ACTIONS = [
    np.array([-1.0]),
    np.array([ 0.5]),
    np.array([-0.5]),
    np.array([ 1.0])
]

def discrete_action(idx):
    return DISCRETE_ACTIONS[idx] 

def pick_random_action():
    return DISCRETE_ACTIONS[random.randint(0,  3)]

# print(DISCRETE_ACTIONS)
# print(len(DISCRETE_ACTIONS))

class RainbowDQNAgent:
    def __init__(self, q_net, targ_net, buffer, optim, gamma =0.9, n_step=5):
        self.q_net=q_net
        self.targ_net=targ_net
        self.buffer=buffer
        self.optim=optim
        self.gamma=gamma
        self.n_step=n_step
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 

    def select_action(self, state_tensor, q_net, epsilon = 0.1):
        if random.random() < epsilon:
            action = pick_random_action()
        else:
            # print(state_tensor.shape)
            q_value = q_net(state_tensor)
            
            argmax_q_value = torch.argmax(q_value).item()
            
            action = discrete_action(argmax_q_value)
        
        return action
    
    # def select_action(self, state):
    #     if random.random() < 0.3:
    #         state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    #         q_value = self.q_net(state)
    #         action_idx = q_value.argmax(dim=1).item()
    #         return DISCRETE_ACTIONS[action_idx]
    #     else:
    #         return random.choice(DISCRETE_ACTIONS)
        
    def learn(self, batch=64):
        if len(self.buffer) < batch:
            return
        
        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(batch_size=batch)
        
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        next_actions = self.q_net(next_states).argmax(dim=1)
        next_q_value = self.targ_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        target_q = rewards + (self.gamma ** self.n_step) * next_q_value * (1-dones)

        td_error = target_q - q_values
        
        new_prior = td_error.abs().detach().cpu().numpy()
        self.buffer.update_priorities(indices, new_prior)

        loss = (weights * td_error.pow(2)).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.reset_noise()


    def reset_noise(self):
        self.q_net.reset_noise()
        self.targ_net.reset_noise()
