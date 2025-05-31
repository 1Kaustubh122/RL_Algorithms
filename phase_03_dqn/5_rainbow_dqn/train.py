import os
import sys
import cv2
import torch
import imageio
import numpy as np
from dm_control import suite
from collections import deque
import matplotlib.pyplot as plt

os.makedirs("Results", exist_ok=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from networks.q_network import RainbowQNetwork
from utils.replay_buffer import RainbowReplayBuffer
from Rainbow_dqn_agent import RainbowDQNAgent, DISCRETE_ACTIONS


env = suite.load('cartpole', 'swingup')
obs_spec = env.observation_spec()
obs_dim = sum(np.prod(v.shape) for v in obs_spec.values())
num_action = len(DISCRETE_ACTIONS)

print(env.action_spec())

print("obs dim", obs_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_net = RainbowQNetwork(obs_shape=obs_dim, num_actions=num_action).to(device)
target_net = RainbowQNetwork(obs_shape=obs_dim, num_actions=num_action).to(device)
target_net.load_state_dict(q_net.state_dict())

buffer_capac = 10000
n_step = 5
replay_buffer = RainbowReplayBuffer(buffer_capac, n_step)

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)

agent = RainbowDQNAgent(q_net, target_net, replay_buffer, optimizer, gamma=0.99, n_step=n_step)

print(DISCRETE_ACTIONS)


def train_rainbow(render):
    num_episodes=100
    batch_size=128
    target_update_freq=20
    max_steps=200
    total_steps = 0
    episode_rewards = []

    for episode in range(num_episodes):
        time_step = env.reset()
        obs = time_step.observation
        # print("env, state", state)
        episode_reward = 0
        
        frames = []  ## FOR Gif
        
        # state = flatten_obs(state)


        for step in range(max_steps):
            state = np.concatenate([obs[k] for k in obs])
            state_tensor = torch.tensor(state, dtype=torch.float32).to(agent.device).unsqueeze(0)
            agent.reset_noise()  

            action = agent.select_action(state_tensor, q_net, epsilon=0.1)
            
            # print(action)
            action_idx = DISCRETE_ACTIONS.index(action)
            
            time_step = env.step(action)
            reward = time_step.reward if time_step.reward is not None else 0.0
            next_obs = time_step.observation
            
            next_state = np.concatenate([next_obs[k] for k in next_obs])
            next_state_tensor = torch.tensor(np.expand_dims(next_state, axis=0), dtype=torch.float32).to(device)
            
            done = time_step.last()
            
            episode_reward += reward
            
            agent.buffer.push(agent.n_step, state_tensor.squeeze(0), 
                              action_idx, reward, 
                              torch.tensor(next_state, dtype=torch.float32), done)

            state = next_state
            total_steps += 1

            agent.learn(batch=batch_size)

            if total_steps % target_update_freq == 0:
                agent.targ_net.load_state_dict(agent.q_net.state_dict())
            
            if render and episode >= (num_episodes - 3):
            # if render:
                rgb_frame = env.physics.render(height=84, width=130, camera_id=0)
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

            if done:
                break
        
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1} Reward: {episode_reward:.2f}")
        
        if episode == num_episodes - 1:
            imageio.mimsave("Results/last_episode.gif", frames, fps=30)
    
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.savefig("Results/training_rewards.png")
    plt.close()

    
    return episode_rewards


def main():
    render = True
    train_rainbow(render)    
    if render:
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
