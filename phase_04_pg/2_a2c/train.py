import os
import sys
import cv2
import torch
import imageio
import numpy as np
import gymnasium as gym
from dm_control import suite
import matplotlib.pyplot as plt

os.makedirs("Results", exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Network.policy_network import PolicyNetwork, ValueNetwork

NUM_EPISODE = 1000
MAX_LEN = 350

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def compute_advantages(returns, value):
    return returns - value

def compute_loss(log_probs, advantage):
    return -(log_probs * advantage).sum()

def compute_value_loss(value, returns):
    return (value - returns) ** 2


def train(render: bool):
    
    ## Gym
    # env = gym.make("Pendulum-v1", render_mode="rgb_array" if render else None)
    # obs_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # is_continous = isinstance(env.action_space, gym.spaces.Box)
    # print("Continuous Action Space:", is_continous)
    
    ## DM
    env = suite.load('cartpole', 'swingup')
    obs_spec = env.observation_spec()
    obs_dim = sum(np.prod(v.shape) for v in obs_spec.values())
    action_spec = env.action_spec()
    action_dim = action_spec.shape[0]
    is_continous = True
    
    policy = PolicyNetwork(obs_dim, action_dim, is_continous=is_continous)
    value_net = ValueNetwork(obs_dim)
    optim = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=1e-3)
    # optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.9)

    eps_rews = []
    best_reward = 0
    best_frames = None
    episode_reward = 0
    
    for eps in range(NUM_EPISODE):
        ## Gym
        # obs, _ = env.reset()
        
        ## DM
        time_step = env.reset()
        obs = time_step.observation
        log_probs, rewards, values, entropies = [], [], [], []
        frames = []

        for _ in range(MAX_LEN):
            
            ## Gym
            # state = torch.from_numpy(obs).float().unsqueeze(0)
            state = np.concatenate([obs[k] for k in obs])
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            raw_action, log_prob, dist, entropy = policy.get_action(state_tensor)
            value = value_net(state_tensor)

            values.append(value.squeeze())
            entropies.append(entropy.mean())

            action = torch.tanh(raw_action)
            # log_prob = dist.log_prob(raw_action).sum(dim=-1)
            # print("old", action)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
            action = action.squeeze(0).detach().numpy()
            # print("new", action)
            
            ## GYM
            # obs, reward, done, truncated, _ = env.step(action.squeeze(0).detach().numpy())

            ## DM
            time_step = env.step(action)
            reward = time_step.reward if time_step.reward is not None else 0.0
            next_obs = time_step.observation
            next_state = np.concatenate([next_obs[k] for k in next_obs])
            next_state_tensor = torch.tensor(np.expand_dims(next_state, axis=0), dtype=torch.float32)
            done = time_step.last()
            state_tensor = next_state_tensor
            

            rewards.append(reward)
            log_probs.append(log_prob)
            
            
            ## GYM
            # if render:
            # if render and eps > (NUM_EPISODE - 3):

            #     rgb_frame = env.render()
            #     frames.append(rgb_frame)
            #     if rgb_frame is not None:
            #         bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            #         bgr_frame = cv2.resize(bgr_frame, (bgr_frame.shape[1], bgr_frame.shape[0]))
            #         cv2.imshow("Live Training", bgr_frame)
            #         cv2.waitKey(1)

            # if done or truncated:
            #     break
            
            
            ## DM
            
            rgb_frame = env.physics.render(height=84, width=130, camera_id=0)
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            scale_factor = 4
            bgr_frame = cv2.resize(
                bgr_frame, 
                (bgr_frame.shape[1] * scale_factor, bgr_frame.shape[0] * scale_factor), 
                interpolation=cv2.INTER_LINEAR
            )
            frames.append(rgb_frame) 
        
       
            # if episode_reward >= best_reward:
            # if render:
            #     cv2.imshow("Live Training", bgr_frame)
            #     cv2.waitKey(1)
            
            
            if done :
                break

        episode_reward = sum(rewards)
        eps_rews.append(sum(rewards))
        print(f"Episode {eps+1} Reward: {episode_reward:.2f}")

        if eps == NUM_EPISODE - 1 and render:
            imageio.mimsave("Results/last_episode.gif", frames, fps=30)

        returns = compute_returns(rewards)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        advantages = compute_advantages(returns, values.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_frames = frames.copy()
        if eps == NUM_EPISODE - 1 and best_frames is not None:
            imageio.mimsave("Results/best_episode.gif", best_frames, fps=30)
        
        policy_loss = compute_loss(log_probs, advantages)
        value_loss = compute_value_loss(values, returns).mean()
        entropy_bonus = entropies.mean()

        total_loss = policy_loss + 0.5 * value_loss - 0.1 * entropy_bonus

        optim.zero_grad()
        total_loss.backward()
        optim.step()
        scheduler.step()

    plt.plot(eps_rews)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.savefig("Results/training_rewards.png")
    plt.close()

    
    
def main():
    render = True
    train(render=render)    
    if render:
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()

