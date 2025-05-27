##   wrappers    ##

import numpy as np
import gymnasium as gym

class NormalizeObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_mean = 0.0
        self.obs_var = 1.0
        self.count = 1e-4
    
    def observation(self, observation):
        self.obs_mean = 0.99 * self.obs_mean + 0.01 * observation
        self.obs_var = 0.99 * self.obs_var + 0.01 * (observation - self.obs_mean) ** 2
        return (observation - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)
    
class ClipAction(gym.ActionWrapper):
    def action(self, action):
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

def make_env(env_id : str):
    env = gym.make(env_id)
    env = ClipAction(env)
    env = NormalizeObs(env)