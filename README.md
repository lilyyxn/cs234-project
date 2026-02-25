ref: https://github.com/Farama-Foundation/gymnasium-env-template
<!--
# Gymnasium Examples
Some simple examples of Gymnasium environments and wrappers.
For some explanations of these examples, see the [Gymnasium documentation](https://gymnasium.farama.org).

### Environments
This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
- `GridWorldEnv`: Simplistic implementation of gridworld environment

### Wrappers
This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).
- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment
-->

## Installation

To install the environment and dependencies:

```shell
pip install -r requirements.txt
pip install -e .
```

## Visualize in pygame before training
The agent takes random action at each state.

```shell
python grid_world_env/run_env.py
```

## Train PPO

```shell
python grid_world_env/train_ppo.py
```

Optional flags:
- `--timesteps 100000` — total training timesteps (default: 100k)
- `--lr 3e-4` — learning rate
- `--gamma 0.99` — discount factor
- `--seed 0` — random seed
- `--save-path ppo_gridworld` — where to save the model
- `--n-envs 4` — number of parallel environments
- `--log-dir logs/ppo_gridworld` — TensorBoard log directory
- `--reward-0-step 1` — per-step reward under func_0 (default: 1)
- `--reward-0-terminal 100` — terminal reward under func_0 (default: 100)
- `--reward-1-step -1` — per-step reward under func_1 (default: -1)
- `--reward-1-terminal 100` — terminal reward under func_1 (default: 100)
- `--mask-actions` — only allow valid moves (no wall bumping). Uses MaskablePPO from sb3-contrib.

Models are auto-saved to `models/` with a name encoding the config (e.g. `r0s1_r0t10_r1s-1_r1t10_masked`).

### Example configurations

Default (agent reaches target):
```shell
python grid_world_env/train_ppo.py
```

Lower terminal reward (agent learns to avoid the target):
```shell
python grid_world_env/train_ppo.py --reward-0-terminal 10 --reward-1-terminal 10
```

Same but with action masking (agent must move, can't wall-bump):
```shell
python grid_world_env/train_ppo.py --reward-0-terminal 10 --reward-1-terminal 10 --mask-actions
```

### View training curves

After training, launch TensorBoard to compare all runs:

```shell
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser.

## Evaluate trained model

Loads a saved model and renders episodes with pygame:

```shell
python grid_world_env/eval_ppo.py
```

Optional flags:
- `--model-path ppo_gridworld` — path to saved model (without .zip)
- `--episodes 5` — number of episodes to visualize
- `--seed 42` — random seed
- `--record` — save episodes as mp4 videos instead of live rendering
- `--video-dir videos` — directory for recorded videos
- `--reward-0-step`, `--reward-0-terminal`, `--reward-1-step`, `--reward-1-terminal` — override reward config (auto-detected from model name)
- `--mask-actions` — enable action masking (auto-detected from model name)

Run with no arguments to list saved models and select one by number.
