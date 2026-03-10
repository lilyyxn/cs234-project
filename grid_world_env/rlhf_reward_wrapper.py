"""
Gymnasium wrapper that applies the learned RLHF correction to the proxy reward.

At each step: reward = R_proxy(s,a,s') + G(s,a,s')

where R_proxy is the original environment reward (+1/step) and G is the learned
correction from the reward model.
"""

import gymnasium as gym
import torch
import numpy as np


class RLHFRewardWrapper(gym.Wrapper):
    """Adds the learned reward correction G(s,a,s') to the proxy reward."""

    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, proxy_reward, terminated, truncated, info = self.env.step(action)

        # Compute G(s, a, s') from the reward model
        with torch.no_grad():
            s = torch.FloatTensor(self._last_obs)
            a_onehot = torch.zeros(self.reward_model.action_dim)
            a_onehot[int(action)] = 1.0
            s_next = torch.FloatTensor(obs)
            correction = self.reward_model(s, a_onehot, s_next).item()

        info["proxy_reward"] = proxy_reward
        info["reward_correction"] = correction

        # R'(s,a,s') = R(s,a,s') + G(s,a,s')
        reward = proxy_reward + correction

        self._last_obs = obs
        return obs, reward, terminated, truncated, info
