"""Parametric reward model that learns the 4 env reward parameters from preferences.

Instead of a neural network, this model learns 4 scalar parameters matching
the GridWorldEnv reward structure exactly:

    R(τ; θ) = r_0_step     * n_steps_phase0
            + r_0_terminal * phase0_terminal
            + r_1_step     * n_steps_phase1
            + r_1_terminal * phase1_terminal

The learned parameters can be plugged directly back into make_env(), so the
corrected reward function lives inside the environment rather than in an
external wrapper.

Training uses the same Bradley-Terry BCE loss as the neural reward model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grid_world_env.rlhf.preference_data import Trajectory


class ParametricRewardModel(nn.Module):
    """4-scalar parametric reward model matching GridWorldEnv reward structure.

    Parameters:
        r_0_step:     reward per step in phase 0
        r_0_terminal: reward at phase 0 completion (transition 0 → 1)
        r_1_step:     reward per step in phase 1
        r_1_terminal: reward at phase 1 completion (episode termination)
    """

    def __init__(self):
        super().__init__()
        self.r_0_step     = nn.Parameter(torch.tensor(0.0))
        self.r_0_terminal = nn.Parameter(torch.tensor(1.0))
        self.r_1_step     = nn.Parameter(torch.tensor(0.0))
        self.r_1_terminal = nn.Parameter(torch.tensor(1.0))

    def trajectory_return(self, features: torch.Tensor) -> torch.Tensor:
        """Compute return for a batch of trajectory feature vectors.

        Args:
            features: (batch, 4) tensor —
                      [n_steps_phase0, phase0_terminal,
                       n_steps_phase1, phase1_terminal]
        Returns:
            (batch,) return estimates.
        """
        return (self.r_0_step     * features[:, 0]
              + self.r_0_terminal * features[:, 1]
              + self.r_1_step     * features[:, 2]
              + self.r_1_terminal * features[:, 3])

    def as_dict(self) -> dict:
        """Return learned parameters as a plain Python dict (detached)."""
        return {
            "r_0_step":     self.r_0_step.item(),
            "r_0_terminal": self.r_0_terminal.item(),
            "r_1_step":     self.r_1_step.item(),
            "r_1_terminal": self.r_1_terminal.item(),
        }


def extract_trajectory_features(traj: Trajectory) -> np.ndarray:
    """Extract 4-dimensional feature vector from a Trajectory.

    Features: [n_steps_phase0, phase0_terminal, n_steps_phase1, phase1_terminal]

    - n_steps_phase0:  steps spent in phase 0
    - phase0_terminal: 1.0 if phase 0 was completed (phase transition occurred)
    - n_steps_phase1:  steps spent in phase 1
    - phase1_terminal: 1.0 if phase 1 was completed (episode terminated, not truncated)
    """
    phases = traj.observations[:, 2]   # obs index 2 = phase (0.0 or 1.0)

    n_steps_phase0 = float(np.sum(phases == 0.0))
    n_steps_phase1 = float(np.sum(phases == 1.0))
    phase0_terminal = 1.0 if n_steps_phase1 > 0 else 0.0
    phase1_terminal = 1.0 if (traj.terminated and n_steps_phase1 > 0) else 0.0

    return np.array(
        [n_steps_phase0, phase0_terminal, n_steps_phase1, phase1_terminal],
        dtype=np.float32,
    )


def train_parametric_reward_model(
    model: ParametricRewardModel,
    preference_pairs: list,
    n_epochs: int = 100,
    lr: float = 1e-2,
) -> ParametricRewardModel:
    """Train parametric reward model on (traj_a, traj_b, label) preference pairs.

    Bradley-Terry BCE loss — identical objective to the neural reward model,
    but optimizing 4 interpretable scalars instead of network weights.

    Args:
        model:            ParametricRewardModel instance (modified in-place)
        preference_pairs: list of (Trajectory, Trajectory, int) with label=1 if τ_a preferred
        n_epochs:         gradient steps over the full dataset
        lr:               Adam learning rate

    Returns:
        Trained model (same object as input).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Pre-extract features once to avoid redundant work per epoch
    feat_pairs = []
    for traj_a, traj_b, label in preference_pairs:
        fa = torch.tensor(extract_trajectory_features(traj_a),
                          dtype=torch.float32).unsqueeze(0)
        fb = torch.tensor(extract_trajectory_features(traj_b),
                          dtype=torch.float32).unsqueeze(0)
        feat_pairs.append((fa, fb, torch.tensor(float(label))))

    for _ in range(n_epochs):
        for fa, fb, label in feat_pairs:
            R_a = model.trajectory_return(fa).squeeze()
            R_b = model.trajectory_return(fb).squeeze()
            loss = F.binary_cross_entropy_with_logits(R_a - R_b, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
