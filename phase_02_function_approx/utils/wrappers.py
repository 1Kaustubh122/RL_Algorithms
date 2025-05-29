import gymnasium as gym
import numpy as np
from collections import deque

class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1.0,
            shape=(shp[0] * k, shp[1], shp[2]),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def observation(self, obs):
        self.frames.append(obs)
        return self._get_ob()

    def _get_ob(self):
        return np.concatenate(list(self.frames), axis=0)

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(NormalizeObservation, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        from cv2 import resize, INTER_AREA
        self.shape = shape
        self.resize = lambda img: resize(img, self.shape, interpolation=INTER_AREA)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.shape[1], self.shape[0], shp[2]),
            dtype=env.observation_space.dtype,
        )

    def observation(self, obs):
        return self.resize(obs)

def make_env(env_name, frame_stack=4, resize_shape=(84, 84)):
    env = gym.make(env_name, render_mode="rgb_array")
    env = ResizeObservation(env, resize_shape)
    env = NormalizeObservation(env)
    env = FrameStack(env, frame_stack)
    return env
