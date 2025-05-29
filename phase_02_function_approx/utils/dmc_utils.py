import cv2
import numpy as np

def preprocess_obs(obs_dict, key="pixels"):
    obs = obs_dict[key]  ## shape (height, width, channels), uint8
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    obs = np.transpose(obs, (2, 0, 1))  ## Convert to (C, H, W)
    return obs.astype(np.uint8)
