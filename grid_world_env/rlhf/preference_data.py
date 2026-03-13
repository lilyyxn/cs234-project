"""Trajectory collection and preference pair generation for RLHF."""
from dataclasses import dataclass

import numpy as np
import torch

from grid_world_env.rlhf.ground_truth import compute_gt_return


@dataclass
class Trajectory:
    observations:  np.ndarray   # (T, obs_dim)
    actions:       np.ndarray   # (T,)
    proxy_rewards: np.ndarray   # (T,)
    terminated:    bool
    gt_return:     float
    proxy_return:  float


def collect_trajectory(policy, env, device: str = "cpu") -> Trajectory:
    """Roll out policy for one episode and return a Trajectory.

    Args:
        policy: ActorCritic instance (with get_action_and_value())
        env:    FullyObservable-wrapped GridWorldEnv
        device: torch device string
    """
    obs_list, action_list, reward_list = [], [], []

    obs, _ = env.reset()
    terminated = False

    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = policy.get_action_and_value(obs_t)
        action_int = action.item()

        next_obs, proxy_reward, term, trunc, _ = env.step(action_int)

        obs_list.append(obs)
        action_list.append(action_int)
        reward_list.append(proxy_reward)

        obs = next_obs
        terminated = bool(term)

        if term or trunc:
            break

    observations  = np.array(obs_list,   dtype=np.float32)
    actions       = np.array(action_list, dtype=np.int64)
    proxy_rewards = np.array(reward_list, dtype=np.float32)

    return Trajectory(
        observations=observations,
        actions=actions,
        proxy_rewards=proxy_rewards,
        terminated=terminated,
        gt_return=compute_gt_return(observations, terminated),
        proxy_return=float(proxy_rewards.sum()),
    )


def collect_trajectories(policy, env, n: int, device: str = "cpu") -> list:
    """Collect n full episodes."""
    return [collect_trajectory(policy, env, device=device) for _ in range(n)]


def boltzmann_label(gt_a: float, gt_b: float,
                    rng: np.random.Generator = None,
                    beta: float = 3.0) -> int:
    """Sample a preference label from the Boltzmann (Bradley-Terry) model.

    P(τ_a > τ_b) = σ(β * (R_GT(τ_a) − R_GT(τ_b)))

    Returns 1 if τ_a is preferred, 0 if τ_b is preferred.
    """
    if rng is None:
        rng = np.random.default_rng()
    p_a_wins = 1.0 / (1.0 + np.exp(-beta * (gt_a - gt_b)))
    return int(rng.random() < p_a_wins)


def generate_preference_pairs(trajectories: list, n_pairs: int,
                               seed: int = 0) -> list:
    """Sample n_pairs of (traj_a, traj_b, label) from a list of trajectories.

    Returns list of (Trajectory, Trajectory, int) where label=1 means τ_a preferred.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    n = len(trajectories)
    for _ in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        traj_a, traj_b = trajectories[i], trajectories[j]
        label = boltzmann_label(traj_a.gt_return, traj_b.gt_return, rng=rng)
        pairs.append((traj_a, traj_b, label))
    return pairs
