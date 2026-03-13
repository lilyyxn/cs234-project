import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class FullyObservable(gym.ObservationWrapper):
    """Wrapper that returns a 4D observation: [dx, dy, phase, t/T].

    This makes the environment fully observable by including the current
    phase (0 or 1) and normalized timestep (t/T) in the observation.
    """

    def __init__(self, env, max_steps=100):
        super().__init__(env)
        self.max_steps = max_steps
        size = env.unwrapped.size
        self.observation_space = Box(
            low=np.array([-(size - 1), -(size - 1), 0.0, 0.0], dtype=np.float32),
            high=np.array([(size - 1), (size - 1), 1.0, 1.0], dtype=np.float32),
        )

    def observation(self, obs):
        base = self.env.unwrapped
        dx = obs["target"][0] - obs["agent"][0]
        dy = obs["target"][1] - obs["agent"][1]
        phase = float(base.current_phase)
        t_norm = float(base._step_count) / self.max_steps
        return np.array([dx, dy, phase, t_norm], dtype=np.float32)
