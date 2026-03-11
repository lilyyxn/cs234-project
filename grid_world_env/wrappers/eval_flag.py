import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class EvalFlagWrapper(gym.ObservationWrapper):
    """Appends a scalar eval_flag in {0, 1} to the observation.

    Wraps a FullyObservable env (4D obs) and returns 5D obs:
      [dx, dy, phase, t/T, eval_flag]

    Call set_eval_flag(True) before RLHF trajectory collection so the
    reward model trains on flag=1 observations. Leave flag=False for
    proxy training and deployment evaluation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._eval_flag = 0.0
        low  = np.append(env.observation_space.low,  0.0).astype(np.float32)
        high = np.append(env.observation_space.high, 1.0).astype(np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def set_eval_flag(self, flag: bool) -> None:
        """Set the eval flag. Persists until changed."""
        self._eval_flag = 1.0 if flag else 0.0

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.append(obs, self._eval_flag).astype(np.float32)
