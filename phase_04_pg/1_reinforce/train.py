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

from Network.policy_network import PolicyNetwork

NUM_EPISODE = 100
MAX_LEN = 500
DISCRETE_ACTIONS = [np.array([v]) for v in np.linspace(-1.0, 1.0, 5)]

print(DISCRETE_ACTIONS)

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def compute_loss(log_probs, returns):
    return -(log_probs * returns).sum()



def train(render : bool):
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
    action_dim = len(DISCRETE_ACTIONS)
    is_continous = False

    lr = 1e-2
    # print(torch.flatten(torch.tensor(obs_dim.shape)))
    # print(action_dim)

    policy = PolicyNetwork(obs_dim, action_dim, is_continous)

    optim = torch.optim.Adam(policy.parameters(), lr)

    eps_rews = []
    best_reward = 0
    best_frames = None

    for eps in range(NUM_EPISODE):
        
        ## Gym
        # obs, _ = env.reset()    
        
        ## DM
        time_step = env.reset()
        obs = time_step.observation
        state = np.concatenate([obs[k] for k in obs])
        state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32)
        
        log_probs, rewards = [], []
        frames = [] ## For GiF
        
        for _ in range(MAX_LEN):
            # state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            # state = torch.from_numpy(obs).float().unsqueeze(0)

            
            action, log_prob, _, _ = policy.get_action(state_tensor)
            ## Gym
            # obs, reward, done, truncated, _ = env.step(action.item())

            ## DM
            time_step = env.step(DISCRETE_ACTIONS[action.item()])
            reward = time_step.reward if time_step.reward is not None else 0.0
            next_obs = time_step.observation
            next_state = np.concatenate([next_obs[k] for k in next_obs])
            next_state_tensor = torch.tensor(np.expand_dims(next_state, axis=0), dtype=torch.float32)
            done = time_step.last()
            state_tensor = next_state_tensor
            
            rewards.append(reward)
            log_probs.append(log_prob)
            
            
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

            if render and eps > (NUM_EPISODE - 3):
            # if render:
                cv2.imshow("Live Training", bgr_frame)
                cv2.waitKey(1)
            
            ## gym
                # rgb_frame = env.render()
                # frames.append(rgb_frame)
                # if rgb_frame is not None:
                #     bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                #     bgr_frame = cv2.resize(
                #         bgr_frame, 
                #         (bgr_frame.shape[1], bgr_frame.shape[0]), 
                #         interpolation=cv2.INTER_LINEAR
                #     )
                #     cv2.imshow("Live Training", bgr_frame)
                #     cv2.waitKey(1)

        
            if done :
                break
        
        episode_reward = sum(rewards)
        eps_rews.append(sum(rewards))
        print(f"Episode {eps+1} Reward: {episode_reward:.2f}")
        
        if eps == NUM_EPISODE - 1:
            imageio.mimsave("Results/last_episode.gif", frames, fps=30)
        
        returns = compute_returns(rewards)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns = returns - returns.mean()
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_frames = frames.copy()
        if eps == NUM_EPISODE - 1 and best_frames is not None:
            imageio.mimsave("Results/best_episode.gif", best_frames, fps=30)
        
        log_probs = torch.stack(log_probs)
        loss = compute_loss(log_probs, returns)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
    
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