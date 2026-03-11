import numpy as np
import pytest


def _make_flag_env():
    import gymnasium
    import grid_world_env
    from grid_world_env.wrappers import FullyObservable
    from grid_world_env.wrappers.eval_flag import EvalFlagWrapper
    base = gymnasium.make(
        "grid_world_env/GridWorld-v0",
        max_episode_steps=100,
    )
    base = FullyObservable(base, max_steps=100)
    return EvalFlagWrapper(base)


def test_eval_flag_obs_shape():
    """Obs shape should be (5,) — 4D base + eval_flag."""
    env = _make_flag_env()
    obs, _ = env.reset()
    assert obs.shape == (5,), f"Expected (5,), got {obs.shape}"


def test_eval_flag_default_zero():
    """eval_flag should be 0.0 by default."""
    env = _make_flag_env()
    obs, _ = env.reset()
    assert obs[4] == 0.0


def test_eval_flag_set_one():
    """After set_eval_flag(True), obs[-1] should be 1.0."""
    env = _make_flag_env()
    env.set_eval_flag(True)
    obs, _ = env.reset()
    assert obs[4] == 1.0


def test_eval_flag_persists_across_steps():
    """Flag should appear in every step obs, not just reset obs."""
    env = _make_flag_env()
    env.set_eval_flag(True)
    env.reset()
    obs, _, _, _, _ = env.step(0)
    assert obs[4] == 1.0


def test_eval_flag_can_be_toggled():
    """set_eval_flag can switch between 0 and 1."""
    env = _make_flag_env()
    env.set_eval_flag(True)
    obs1, _ = env.reset()
    assert obs1[4] == 1.0
    env.set_eval_flag(False)
    obs2, _ = env.reset()
    assert obs2[4] == 0.0


def test_eval_flag_obs_space_shape():
    """observation_space.shape should be (5,)."""
    env = _make_flag_env()
    assert env.observation_space.shape == (5,)


def test_eval_flag_obs_within_bounds():
    """All obs values should be within observation_space bounds."""
    env = _make_flag_env()
    env.set_eval_flag(True)
    obs, _ = env.reset()
    assert np.all(obs >= env.observation_space.low)
    assert np.all(obs <= env.observation_space.high)
