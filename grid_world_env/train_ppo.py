import argparse
import os
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

import grid_world_env
from grid_world_env.wrappers import RelativePosition


def make_env(render_mode=None, reward_0_step=1, reward_0_terminal=100,
             reward_1_step=-1, reward_1_terminal=100, mask_actions=False,
             reward_mode="default", loop_detection=False, loop_window=10,
             loop_grace_period=1):
    env = gymnasium.make(
        "grid_world_env/GridWorld-v0",
        max_episode_steps=100,
        render_mode=render_mode,
        reward_0_step=reward_0_step,
        reward_0_terminal=reward_0_terminal,
        reward_1_step=reward_1_step,
        reward_1_terminal=reward_1_terminal,
        reward_mode=reward_mode,
        loop_detection=loop_detection,
        loop_window=loop_window,
        loop_grace_period=loop_grace_period,
    )
    if mask_actions:
        env = ActionMasker(env, lambda e: e.get_wrapper_attr("action_masks")())
    env = RelativePosition(env)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train PPO on GridWorld")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save model (auto-generated if not set)")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory (auto-generated if not set)")
    parser.add_argument("--reward-0-step", type=float, default=1, help="Per-step reward for func_0")
    parser.add_argument("--reward-0-terminal", type=float, default=100, help="Terminal reward for func_0")
    parser.add_argument("--reward-1-step", type=float, default=-1, help="Per-step reward for func_1")
    parser.add_argument("--reward-1-terminal", type=float, default=100, help="Terminal reward for func_1")
    parser.add_argument("--mask-actions", action="store_true", help="Enable action masking (no wall bumping)")
    parser.add_argument("--relative-reward", action="store_true", help="Use relative position (negative Manhattan distance) as reward instead of +1/-1")
    parser.add_argument("--loop-detection", action="store_true", help="Switch reward function when looping behavior is detected")
    parser.add_argument("--loop-window", type=int, default=10, help="Number of recent steps to check for looping (default: 10)")
    parser.add_argument("--loop-grace-period", type=int, default=5, help="Number of loop detections to tolerate before switching reward function (default: 5)")
    args = parser.parse_args()

    # Auto-generate run name from reward config (use ints when possible for clean filenames)
    def fmt(v):
        return str(int(v)) if v == int(v) else str(v)
    run_name = f"r0s{fmt(args.reward_0_step)}_r0t{fmt(args.reward_0_terminal)}_r1s{fmt(args.reward_1_step)}_r1t{fmt(args.reward_1_terminal)}"
    if args.mask_actions:
        run_name += "_masked"
    if args.relative_reward:
        run_name += "_relrew"
    if args.loop_detection:
        run_name += f"_loop{args.loop_window}"
    if args.save_path is None:
        args.save_path = f"models/{run_name}"
    if args.log_dir is None:
        args.log_dir = f"logs/{run_name}"

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    reward_mode = "relative" if args.relative_reward else "default"
    print(f"Using reward mode: {reward_mode}")
    env_kwargs = dict(
        reward_0_step=args.reward_0_step,
        reward_0_terminal=args.reward_0_terminal,
        reward_1_step=args.reward_1_step,
        reward_1_terminal=args.reward_1_terminal,
        mask_actions=args.mask_actions,
        reward_mode=reward_mode,
        loop_detection=args.loop_detection,
        loop_window=args.loop_window,
        loop_grace_period=args.loop_grace_period,
    )
    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed, env_kwargs=env_kwargs)

    if args.mask_actions:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            verbose=1,
            seed=args.seed,
            tensorboard_log=args.log_dir,
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            verbose=1,
            seed=args.seed,
            tensorboard_log=args.log_dir,
        )

    print(f"Training {'MaskablePPO' if args.mask_actions else 'PPO'} for {args.timesteps} timesteps...")
    print(f"TensorBoard logs: {args.log_dir}")
    print("  View with: tensorboard --logdir", args.log_dir)
    model.learn(total_timesteps=args.timesteps)

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}.zip")


if __name__ == "__main__":
    main()
