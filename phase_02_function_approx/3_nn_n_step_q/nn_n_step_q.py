import os
import sys
import cv2
import torch
import random
import imageio
import numpy as np
from dm_control import suite
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dm_control.suite.wrappers import pixels

os.makedirs("Results", exist_ok=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from networks.cnn import QNetwork
from utils.replay_buffer import ReplayBuffer, NStepTransitionBuffer

env = suite.load(domain_name="cartpole", task_name="swingup")

observation_spec = env.observation_spec()
## OrderedDict([('position', Array(shape=(3,), dtype=dtype('float64'), name='position')), 
## ('velocity', Array(shape=(2,), dtype=dtype('float64'), name='velocity'))])

action_spec = env.observation_spec()
## BoundedArray(shape=(1,), dtype=dtype('float64'), name=None, minimum=[-1.], maximum=[1.])

time_step = env.reset()
obs = time_step.observation

state = np.concatenate([obs[k] for k in obs])

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

action = pick_random_action() 

x = env.step(action)
print(x)
# net = QNetwork(obs_shape=5, num_actions=7)

# combined = np.concatenate((x.observation['position'], x.observation['velocity']))

# dummy_state = torch.tensor(np.expand_dims(combined, axis=0), dtype=torch.float32)
# print(f"Dummy state shape {dummy_state.shape}")
# q_values = net(dummy_state)
# print(q_values)
# print(q_values.shape)  ## torch.Size([1, 7])


def compute_loss(net, target_net, states, actions, retuens, next_states, dones, gamma, n=3):
    states = states.view(states.size(0), -1)
    next_states = next_states.view(next_states.size(0), -1)

    q_value = net(states)  
    state_action_value = q_value.gather(1, actions.unsqueeze(1)).squeeze(1)  

    with torch.no_grad():
        next_q_value = target_net(next_states)  
        max_next_q_value = next_q_value.max(1)[0] 
        target_q_values = retuens + (gamma ** n) * max_next_q_value * (1 - dones.float())

    loss = F.mse_loss(state_action_value, target_q_values)

    return loss




obs_dim = 5
num_action = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_net = QNetwork(obs_shape=obs_dim, num_actions=num_action).to(device)
target_net = QNetwork(obs_shape=obs_dim, num_actions=num_action).to(device)
target_net.load_state_dict(q_net.state_dict())  ## Sync weights

print(f"using device {device}")

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)

replay_buffer = ReplayBuffer(capacity=5000, device=device)
n_step_buffer = NStepTransitionBuffer(n=3, gamma=0.99, main_buffer=replay_buffer)


def e_greedy(state_tensor, q_net, epsilon = 0.1):
    if random.random() < epsilon:
        action = pick_random_action()
    else:
        # print(state_tensor.shape)
        q_value = q_net(state_tensor)
        
        argmax_q_value = torch.argmax(q_value).item()
        
        action = discrete_action(argmax_q_value)
    
    return action


def train(render):
    num_eps = 500
    batch_size = 64
    target_update_freq = 10
    save_model_freq = 50
    
    reward_log = []

    for episode in range(num_eps):
        time_step = env.reset()
        obs = time_step.observation
        state = np.concatenate([obs[k] for k in obs])
        state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32).to(device)
        
        episode_reward = 0
        frames = []  ## FOR Gif
        
        for t in range(EPISODE_MAX_LEN):
            action = e_greedy(state_tensor, q_net, epsilon=0.1)
            
            time_step = env.step(action)
            reward = time_step.reward
            next_obs = time_step.observation
            next_state = np.concatenate([next_obs[k] for k in next_obs])
            next_state_tensor = torch.tensor(np.expand_dims(next_state, axis=0), dtype=torch.float32).to(device)
            done = time_step.last()

            action_idx = DISCRETE_ACTIONS.index(action)
            
            n_step_buffer.push(state_tensor, action_idx, reward, next_state_tensor, done)

            if done == True:
                n_step_buffer._flush(final=True)

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones, _ = replay_buffer.sample(batch_size)
                loss = compute_loss(q_net, target_net, states, actions, rewards, next_states, dones, gamma=0.99, n=3)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state_tensor = next_state_tensor

            episode_reward += reward
            
            if render:
                rgb_frame = env.physics.render(height=84, width=84, camera_id=0)
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                scale_factor = 4
                bgr_frame = cv2.resize(
                    bgr_frame, 
                    (bgr_frame.shape[1] * scale_factor, bgr_frame.shape[0] * scale_factor), 
                    interpolation=cv2.INTER_LINEAR
                )
                cv2.imshow("Live Training", bgr_frame)
                cv2.waitKey(1)
                frames.append(rgb_frame)
            
        reward_log.append(episode_reward)
        print(f"Episode {episode} — Reward: {episode_reward:.2f}")

        if episode % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        if episode % save_model_freq == 0:
            torch.save(q_net.state_dict(), f"q_net_episode_{episode}.pth")

        if episode == num_eps - 1:
            imageio.mimsave("Results/last_episode.gif", frames, fps=30)
    plt.plot(reward_log)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.savefig("Results/training_rewards.png")
    plt.close()



def main():
    render = True
    train(render)
    
    if render:
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
