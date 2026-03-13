"""Ground-truth reward for the two-phase GridWorld.

Ground truth (what the human actually wants):
  +1.0  when phase-0 is completed (obs[2] transitions 0→1)
  +1.0  when phase-1 is completed (episode terminates, not truncated)
   0.0  otherwise

Maximum possible GT return per episode = 2.0.
"""
import numpy as np


def compute_gt_return(observations: np.ndarray, terminated: bool) -> float:
    """Compute the ground-truth return for a trajectory.

    Args:
        observations: shape (T, 4) array of FullyObservable obs [dx,dy,phase,t/T]
        terminated:   True if episode ended via phase-1 completion (not truncation)

    Returns:
        Ground-truth return in {0.0, 1.0, 2.0}.
    """
    gt = 0.0
    for t in range(1, len(observations)):
        if observations[t, 2] > observations[t - 1, 2]:   # phase 0→1 transition
            gt += 1.0
    if terminated:
        gt += 1.0
    return gt
