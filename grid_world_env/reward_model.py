"""
Reward model for RLHF: learns a correction term G(s, a, s') from preferences.

The updated reward is R'(s,a,s') = R(s,a,s') + G(s,a,s'), where R is the proxy
reward and G is learned from pairwise preferences using the Bradley-Terry model.

References:
- Christiano et al. 2017, "Deep RL from Human Preferences" (Section 2.2.3)
- Lee et al. 2021, "B-Pref" (Section 3.3, Algorithm 1)
"""

import torch
import torch.nn as nn
import numpy as np


class RewardModel(nn.Module):
    """Neural network that predicts a scalar reward correction G(s, a, s').

    Input: concatenation of [obs (4), action_onehot (4), next_obs (4)] = 12 dims.
    Output: scalar reward correction.
    """

    def __init__(self, obs_dim=4, action_dim=4, hidden=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action_onehot, next_obs):
        """Predict G(s, a, s') for a single transition or batch."""
        x = torch.cat([obs, action_onehot, next_obs], dim=-1)
        return self.net(x)

    def segment_return(self, segment):
        """Compute sum of predicted rewards over a trajectory segment.

        Args:
            segment: list of (obs, action, next_obs) tuples,
                     where obs/next_obs are numpy arrays of shape (4,)
                     and action is an int in {0,1,2,3}.
        Returns:
            Scalar tensor: sum of G(s,a,s') over the segment.
        """
        total = torch.tensor(0.0)
        for (obs, action, next_obs) in segment:
            obs_t = torch.FloatTensor(obs)
            a_onehot = torch.zeros(self.action_dim)
            a_onehot[int(action)] = 1.0
            next_obs_t = torch.FloatTensor(next_obs)
            total = total + self.forward(obs_t, a_onehot, next_obs_t).squeeze()
        return total


def train_reward_model(reward_model, preference_dataset, epochs=50, lr=1e-3):
    """Train the reward model on preference data using Bradley-Terry cross-entropy loss.

    Args:
        reward_model: RewardModel instance.
        preference_dataset: list of (segment_1, segment_2, (mu_1, mu_2)) tuples.
        epochs: number of training epochs.
        lr: learning rate.

    Returns:
        List of per-epoch average losses.
    """
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        np.random.shuffle(preference_dataset)

        for (seg1, seg2, (mu1, mu2)) in preference_dataset:
            r1 = reward_model.segment_return(seg1)
            r2 = reward_model.segment_return(seg2)

            # Bradley-Terry log-probabilities
            log_p1 = r1 - torch.logaddexp(r1, r2)
            log_p2 = r2 - torch.logaddexp(r1, r2)

            # Cross-entropy loss
            loss = -(mu1 * log_p1 + mu2 * log_p2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(preference_dataset), 1)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"  Reward model epoch {epoch}/{epochs}, loss: {avg_loss:.4f}")

    return losses
