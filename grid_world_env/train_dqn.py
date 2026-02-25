import argparse
import os
import gymnasium
from stable_baselines3 import DQN

import grid_world_env
from grid_world_env.wrappers import RelativePosition


def make_env(render_mode=None, reward_0_step=1, reward_0_terminal=100,
             reward_1_step=-1, reward_1_terminal=100):
    env = gymnasium.make(
        "grid_world_env/GridWorld-v0",
        max_episode_steps=100,
        render_mode=render_mode,
        reward_0_step=reward_0_step,
        reward_0_terminal=reward_0_terminal,
        reward_1_step=reward_1_step,
        reward_1_terminal=reward_1_terminal,
    )
    env = RelativePosition(env)
    return env


def make_run_name(reward_0_step, reward_0_terminal, reward_1_step, reward_1_terminal):
    def fmt(v):
        return str(int(v)) if v == int(v) else str(v)
    return (
        f"dqn_r0s{fmt(reward_0_step)}_r0t{fmt(reward_0_terminal)}"
        f"_r1s{fmt(reward_1_step)}_r1t{fmt(reward_1_terminal)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train DQN on GridWorld")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save model (auto-generated if not set)")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory (auto-generated if not set)")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="Replay buffer capacity")
    parser.add_argument("--learning-starts", type=int, default=1_000, help="Steps before first gradient update")
    parser.add_argument("--exploration-fraction", type=float, default=0.1, help="Fraction of training for epsilon decay")
    parser.add_argument("--exploration-final-eps", type=float, default=0.05, help="Final exploration epsilon")
    parser.add_argument("--target-update-interval", type=int, default=500, help="Steps between target network syncs")
    parser.add_argument("--train-freq", type=int, default=4, help="Environment steps per gradient update")
    parser.add_argument("--reward-0-step", type=float, default=1, help="Per-step reward for func_0")
    parser.add_argument("--reward-0-terminal", type=float, default=100, help="Terminal reward for func_0")
    parser.add_argument("--reward-1-step", type=float, default=-1, help="Per-step reward for func_1")
    parser.add_argument("--reward-1-terminal", type=float, default=100, help="Terminal reward for func_1")
    args = parser.parse_args()

    run_name = make_run_name(
        reward_0_step=args.reward_0_step,
        reward_0_terminal=args.reward_0_terminal,
        reward_1_step=args.reward_1_step,
        reward_1_terminal=args.reward_1_terminal,
    )
    if args.save_path is None:
        args.save_path = f"models/{run_name}"
    if args.log_dir is None:
        args.log_dir = f"logs/{run_name}"

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    env = make_env(
        reward_0_step=args.reward_0_step,
        reward_0_terminal=args.reward_0_terminal,
        reward_1_step=args.reward_1_step,
        reward_1_terminal=args.reward_1_terminal,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        target_update_interval=args.target_update_interval,
        train_freq=args.train_freq,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
    )

    print(f"Training DQN for {args.timesteps} timesteps...")
    print(f"TensorBoard logs: {args.log_dir}")
    print("  View with: tensorboard --logdir", args.log_dir)
    model.learn(total_timesteps=args.timesteps)

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}.zip")


if __name__ == "__main__":
    main()
