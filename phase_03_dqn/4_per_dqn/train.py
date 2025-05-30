import os
import sys
import cv2
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from dm_control import suite

os.makedirs("Results", exist_ok=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.replay_buffer import PriortizedReplayBuffer
from networks.q_network import DuelingQNetwork
from per_dqn_agent import PerDqnAgent, DISCRETE_ACTIONS


EPISODE_MAX_LEN = 300

env = suite.load('cartpole', 'swingup')
obs_spec = env.observation_spec()
obs_dim = sum(np.prod(v.shape) for v in obs_spec.values())
num_action = len(DISCRETE_ACTIONS)
    
print(obs_dim, num_action)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_net = DuelingQNetwork(obs_shape=obs_dim, num_actions=num_action).to(device)
target_net = DuelingQNetwork(obs_shape=obs_dim, num_actions=num_action).to(device)

print(f"using device {device}")

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)

replay_buffer = PriortizedReplayBuffer(capacity=50000)

def train(render):
    target_update_freq = 20
    dqn_agent = PerDqnAgent(q_net, target_net, replay_buffer, optimizer, 
                     epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, gamma=0.99, 
                     target_update_freq=target_update_freq)
    num_eps = 1001
    save_model_freq = 200
    
    reward_log = []

    for episode in range(num_eps):
        time_step = env.reset()
        obs = time_step.observation
        state = np.concatenate([obs[k] for k in obs])
        state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32).to(device)
        
        episode_reward = 0
        frames = []  ## FOR Gif
        
        for t in range(EPISODE_MAX_LEN):
            action, action_idx = dqn_agent.e_greedy_action_select(state_tensor)
            # print(action, action_idx)
            time_step = env.step(action)
            reward = time_step.reward if time_step.reward is not None else 0.0
            next_obs = time_step.observation
            next_state = np.concatenate([next_obs[k] for k in next_obs])
            next_state_tensor = torch.tensor(np.expand_dims(next_state, axis=0), dtype=torch.float32).to(device)
            done = time_step.last()
            
            replay_buffer.push(state_tensor.squeeze(0), action_idx, reward, next_state_tensor.squeeze(0), done)

            dqn_agent.learn()

            episode_reward += reward
            
            if render and episode >= 999:
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

        dqn_agent.update_epsilon()   
        reward_log.append(episode_reward)
        print(f"Episode {episode} — Reward: {episode_reward:.2f}")


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

