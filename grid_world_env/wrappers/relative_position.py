import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class RelativePosition(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(shape=(4,), low=-np.inf, high=np.inf)

    def observation(self, obs):
        rel = obs["target"] - obs["agent"]
        phase = np.array([obs["phase"]], dtype=np.float64)
        timestep = obs["timestep"]
        return np.concatenate([rel, phase, timestep])
