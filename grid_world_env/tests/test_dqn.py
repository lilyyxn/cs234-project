import os
import pytest
from gymnasium.spaces import Discrete

import grid_world_env


def test_make_env_observation_space():
    """make_env returns an env with 2D relative position observation."""
    from grid_world_env.train_dqn import make_env
    env = make_env()
    assert env.observation_space.shape == (2,)


def test_make_env_action_space():
    """make_env returns an env with 4 discrete actions."""
    from grid_world_env.train_dqn import make_env
    env = make_env()
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == 4


def test_dqn_model_initializes():
    """DQN model can be created with the gridworld environment."""
    from stable_baselines3 import DQN
    from grid_world_env.train_dqn import make_env
    env = make_env()
    model = DQN("MlpPolicy", env, verbose=0)
    assert model is not None


def test_dqn_trains_short():
    """DQN can train for a small number of timesteps and predict valid actions."""
    from stable_baselines3 import DQN
    from grid_world_env.train_dqn import make_env
    env = make_env()
    model = DQN("MlpPolicy", env, verbose=0, learning_starts=100)
    model.learn(total_timesteps=200)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(int(action))


def test_dqn_save_load(tmp_path):
    """DQN model can be saved and loaded, and still predicts valid actions."""
    from stable_baselines3 import DQN
    from grid_world_env.train_dqn import make_env
    env = make_env()
    model = DQN("MlpPolicy", env, verbose=0, learning_starts=100)
    model.learn(total_timesteps=200)
    save_path = str(tmp_path / "test_dqn_model")
    model.save(save_path)
    loaded = DQN.load(save_path)
    obs, _ = env.reset()
    action, _ = loaded.predict(obs, deterministic=True)
    assert env.action_space.contains(int(action))


def test_run_name_format():
    """make_run_name encodes reward config with dqn_ prefix."""
    from grid_world_env.train_dqn import make_run_name
    name = make_run_name(reward_0_step=1, reward_0_terminal=100,
                         reward_1_step=-1, reward_1_terminal=100)
    assert name == "dqn_r0s1_r0t100_r1s-1_r1t100"


def test_run_name_float_values():
    """make_run_name uses integer formatting when value is a whole number."""
    from grid_world_env.train_dqn import make_run_name
    name = make_run_name(reward_0_step=0.5, reward_0_terminal=10,
                         reward_1_step=-1, reward_1_terminal=10)
    assert name == "dqn_r0s0.5_r0t10_r1s-1_r1t10"


def test_parse_config_from_path():
    """parse_config_from_path extracts reward config from dqn model filename."""
    from grid_world_env.eval_dqn import parse_config_from_path
    config = parse_config_from_path("models/dqn_r0s1_r0t10_r1s-1_r1t10")
    assert config["reward_0_step"] == 1.0
    assert config["reward_0_terminal"] == 10.0
    assert config["reward_1_step"] == -1.0
    assert config["reward_1_terminal"] == 10.0


def test_run_episodes_count():
    """run_episodes returns a list with one reward per episode."""
    from stable_baselines3 import DQN
    from grid_world_env.train_dqn import make_env
    from grid_world_env.eval_dqn import run_episodes
    env = make_env()
    model = DQN("MlpPolicy", env, verbose=0, learning_starts=100)
    model.learn(total_timesteps=200)
    rewards = run_episodes(model, env, n_episodes=3, seed=0)
    assert len(rewards) == 3
    assert all(isinstance(r, float) for r in rewards)
