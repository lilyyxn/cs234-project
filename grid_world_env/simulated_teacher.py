"""
Stochastic simulated teacher for RLHF preference elicitation.

Generates preference labels using the ground-truth reward function, following
the stochastic teacher model from B-Pref (Lee et al. 2021, Eq. 1):

    P(sigma_a > sigma_b) = sigmoid(r_true(sigma_a) - r_true(sigma_b))

This models a noisy human who is more likely to prefer the better trajectory
but can make mistakes, unlike a deterministic oracle.
"""

import numpy as np


def ground_truth_return(trajectory, reward_step=-1, reward_terminal=100, grid_size=5):
    """Compute return of a trajectory under the ground-truth reward function.

    Ground truth: -1 per step, +100 for reaching the target.
    This is the reward the agent never sees — it only sees the proxy (+1/step).

    Args:
        trajectory: list of (obs, action, next_obs) tuples.
            obs/next_obs are the 4D RelativePosition vectors:
            [rel_x, rel_y, phase, timestep].
        reward_step: per-step reward under ground truth (default: -1).
        reward_terminal: reward for reaching target (default: +100).

    Returns:
        Float: total return under ground-truth reward.
    """
    total = 0.0
    for (obs, action, next_obs) in trajectory:
        # Check if agent reached target: relative position [0, 0] means at target
        at_target = (next_obs[0] == 0.0 and next_obs[1] == 0.0)
        if at_target:
            total += reward_terminal
        else:
            total += reward_step
    return total


def sample_preference(traj_a, traj_b, reward_step=-1, reward_terminal=100):
    """Sample a stochastic preference label for a trajectory pair.

    Uses the Bradley-Terry model (sigmoid of return difference) to sample
    a noisy preference, modeling a realistic human teacher.

    P(a > b) = sigmoid(r_true(a) - r_true(b))

    Args:
        traj_a, traj_b: trajectory lists of (obs, action, next_obs).
        reward_step: ground-truth per-step reward.
        reward_terminal: ground-truth terminal reward.

    Returns:
        (mu_a, mu_b): preference label, either (1,0) or (0,1).
    """
    r_a = ground_truth_return(traj_a, reward_step, reward_terminal)
    r_b = ground_truth_return(traj_b, reward_step, reward_terminal)

    # Sigmoid: P(a preferred) = 1 / (1 + exp(-(r_a - r_b)))
    p_a = 1.0 / (1.0 + np.exp(-(r_a - r_b)))

    # Sample from Bernoulli
    if np.random.random() < p_a:
        return (1.0, 0.0)
    else:
        return (0.0, 1.0)


def build_preference_dataset(trajectories, n_pairs=500, reward_step=-1,
                              reward_terminal=100):
    """Build a preference dataset by sampling pairs and labeling them.

    Args:
        trajectories: list of trajectory lists.
        n_pairs: number of preference pairs to generate.
        reward_step: ground-truth per-step reward.
        reward_terminal: ground-truth terminal reward.

    Returns:
        List of (traj_a, traj_b, (mu_a, mu_b)) tuples.
    """
    n = len(trajectories)
    dataset = []

    for _ in range(n_pairs):
        i, j = np.random.choice(n, size=2, replace=False)
        mu = sample_preference(trajectories[i], trajectories[j],
                               reward_step, reward_terminal)
        dataset.append((trajectories[i], trajectories[j], mu))

    return dataset
