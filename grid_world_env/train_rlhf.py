"""
RLHF training pipeline for GridWorld.

Demonstrates reinforcement learning from human feedback in a toy setting:
1. Train initial policy with proxy reward (+1/step)
2. Collect trajectories from that policy
3. Simulated teacher generates stochastic preferences using ground-truth reward
4. Train reward model G from preferences (Bradley-Terry)
5. Retrain policy with R' = R_proxy + G
6. Repeat for multiple rounds

The environment stays in Phase 0 the entire time — no phase transition.
The agent only ever sees the proxy reward + learned correction.
The ground-truth reward is only used by the simulated teacher.
"""

import argparse
import os
import json
import numpy as np
import torch
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import grid_world_env
from grid_world_env.wrappers import RelativePosition
from grid_world_env.reward_model import RewardModel, train_reward_model
from grid_world_env.simulated_teacher import build_preference_dataset, ground_truth_return
from grid_world_env.rlhf_reward_wrapper import RLHFRewardWrapper


def make_proxy_env(render_mode=None, max_episode_steps=100,
                   reward_0_step=1, reward_0_terminal=100):
    """Create a GridWorld env with proxy reward only (no phase transition).

    The env uses reward_func_0 throughout. We set reward_1 = reward_0 so that
    even if the agent reaches the target (triggering the phase switch internally),
    the reward structure stays the same: +1/step, +100 at terminal.
    """
    env = gymnasium.make(
        "grid_world_env/GridWorld-v0",
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        reward_0_step=reward_0_step,
        reward_0_terminal=reward_0_terminal,
        reward_1_step=reward_0_step,
        reward_1_terminal=reward_0_terminal,
        reward_mode="default",
        loop_detection=False,
        max_steps=max_episode_steps,
        single_goal=True,
    )
    env = RelativePosition(env)
    return env


def make_rlhf_env(reward_model, render_mode=None, max_episode_steps=100,
                  reward_0_step=1, reward_0_terminal=100):
    """Create a GridWorld env wrapped with the RLHF reward correction."""
    env = make_proxy_env(render_mode=render_mode,
                         max_episode_steps=max_episode_steps,
                         reward_0_step=reward_0_step,
                         reward_0_terminal=reward_0_terminal)
    env = RLHFRewardWrapper(env, reward_model)
    return env


def collect_trajectories(model, env_fn, n_episodes=500):
    """Roll out the policy and collect full episode trajectories.

    Args:
        model: trained SB3 model.
        env_fn: callable that returns a single env.
        n_episodes: number of episodes to collect.

    Returns:
        List of trajectories, where each trajectory is a list of
        (obs, action, next_obs) tuples.
    """
    env = env_fn()
    trajectories = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        trajectory = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            action = int(action)
            next_obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append((obs.copy(), action, next_obs.copy()))
            obs = next_obs
            done = terminated or truncated
        trajectories.append(trajectory)

    env.close()
    return trajectories


def evaluate_policy_ground_truth(model, env_fn, n_episodes=100,
                                  reward_step=-1, reward_terminal=100):
    """Evaluate a policy using the ground-truth reward (not the proxy).

    Returns mean and std of ground-truth returns.
    """
    trajectories = collect_trajectories(model, env_fn, n_episodes)
    returns = [ground_truth_return(traj, reward_step, reward_terminal)
               for traj in trajectories]
    return np.mean(returns), np.std(returns), trajectories


def main():
    parser = argparse.ArgumentParser(description="RLHF training on GridWorld")
    # RLHF pipeline params
    parser.add_argument("--n-rounds", type=int, default=3,
                        help="Number of RLHF rounds (collect prefs, train G, retrain policy)")
    parser.add_argument("--n-preferences", type=int, default=500,
                        help="Number of preference pairs per round")
    parser.add_argument("--n-trajectories", type=int, default=500,
                        help="Number of trajectories to collect per round")
    parser.add_argument("--reward-model-epochs", type=int, default=50,
                        help="Epochs to train reward model per round")
    parser.add_argument("--reward-model-lr", type=float, default=1e-3,
                        help="Reward model learning rate")
    parser.add_argument("--reward-model-hidden", type=int, default=64,
                        help="Reward model hidden layer size")

    # PPO params
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="PPO training timesteps per round")
    parser.add_argument("--lr", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout steps")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=1.0, help="GAE lambda")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy coefficient")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=100, help="Max episode steps")

    # Ground-truth reward (used by simulated teacher only)
    parser.add_argument("--gt-reward-step", type=float, default=-1,
                        help="Ground-truth per-step reward (teacher only)")
    parser.add_argument("--gt-reward-terminal", type=float, default=100,
                        help="Ground-truth terminal reward (teacher only)")

    # Proxy reward (what the agent actually sees)
    parser.add_argument("--proxy-reward-step", type=float, default=1,
                        help="Proxy per-step reward")
    parser.add_argument("--proxy-reward-terminal", type=float, default=100,
                        help="Proxy terminal reward")

    # Output
    parser.add_argument("--save-dir", type=str, default="models/rlhf",
                        help="Directory to save models and results")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Env factory functions ---
    def proxy_env_fn():
        return make_proxy_env(max_episode_steps=args.max_steps,
                              reward_0_step=args.proxy_reward_step,
                              reward_0_terminal=args.proxy_reward_terminal)

    def proxy_vec_env_fn():
        return make_vec_env(
            lambda: proxy_env_fn(),
            n_envs=args.n_envs, seed=args.seed,
        )

    # --- Phase 0: Train initial policy with proxy reward ---
    print("=" * 60)
    print("Phase 0: Training initial policy with proxy reward (+1/step)")
    print("=" * 60)

    vec_env = proxy_vec_env_fn()
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef, verbose=1, seed=args.seed,
        tensorboard_log=os.path.join(args.save_dir, "logs", "round_0_proxy"),
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(args.save_dir, "ppo_proxy"))

    # Evaluate proxy policy under ground truth
    gt_mean, gt_std, _ = evaluate_policy_ground_truth(
        model, proxy_env_fn, n_episodes=100,
        reward_step=args.gt_reward_step, reward_terminal=args.gt_reward_terminal,
    )
    print(f"\nProxy policy ground-truth return: {gt_mean:.1f} +/- {gt_std:.1f}")

    results = [{
        "round": 0,
        "type": "proxy",
        "gt_return_mean": float(gt_mean),
        "gt_return_std": float(gt_std),
    }]

    # --- RLHF Rounds ---
    reward_model = RewardModel(hidden=args.reward_model_hidden)

    for round_i in range(1, args.n_rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"RLHF Round {round_i}/{args.n_rounds}")
        print(f"{'=' * 60}")

        # Step 1: Collect trajectories from current policy
        print(f"\n  Collecting {args.n_trajectories} trajectories...")
        trajectories = collect_trajectories(model, proxy_env_fn, args.n_trajectories)

        avg_len = np.mean([len(t) for t in trajectories])
        n_reached = sum(1 for t in trajectories
                        if any(s_next[0] == 0.0 and s_next[1] == 0.0
                               for (_, _, s_next) in t))
        print(f"  Avg trajectory length: {avg_len:.1f}, "
              f"reached goal: {n_reached}/{args.n_trajectories}")

        # Step 2: Build preference dataset from simulated teacher
        print(f"\n  Generating {args.n_preferences} preference pairs (stochastic teacher)...")
        preference_dataset = build_preference_dataset(
            trajectories, n_pairs=args.n_preferences,
            reward_step=args.gt_reward_step,
            reward_terminal=args.gt_reward_terminal,
        )

        # Step 3: Train reward model G
        print(f"\n  Training reward model G ({args.reward_model_epochs} epochs)...")
        rm_losses = train_reward_model(
            reward_model, preference_dataset,
            epochs=args.reward_model_epochs, lr=args.reward_model_lr,
        )

        torch.save(reward_model.state_dict(),
                    os.path.join(args.save_dir, f"reward_model_round{round_i}.pt"))

        # Step 4: Retrain PPO with R' = R_proxy + G
        print(f"\n  Retraining PPO with corrected reward R' = R + G...")

        # Need a fresh reward model reference for the wrapper (same weights)
        def rlhf_env_fn():
            return make_rlhf_env(reward_model,
                                 max_episode_steps=args.max_steps,
                                 reward_0_step=args.proxy_reward_step,
                                 reward_0_terminal=args.proxy_reward_terminal)

        vec_env = make_vec_env(
            lambda: rlhf_env_fn(),
            n_envs=args.n_envs, seed=args.seed,
        )

        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
            n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef, verbose=1, seed=args.seed,
            tensorboard_log=os.path.join(args.save_dir, "logs", f"round_{round_i}_rlhf"),
        )
        model.learn(total_timesteps=args.timesteps)
        model.save(os.path.join(args.save_dir, f"ppo_rlhf_round{round_i}"))

        # Evaluate under ground truth (using proxy env, no correction — pure policy quality)
        gt_mean, gt_std, _ = evaluate_policy_ground_truth(
            model, proxy_env_fn, n_episodes=100,
            reward_step=args.gt_reward_step, reward_terminal=args.gt_reward_terminal,
        )
        print(f"\n  Round {round_i} ground-truth return: {gt_mean:.1f} +/- {gt_std:.1f}")

        results.append({
            "round": round_i,
            "type": "rlhf",
            "gt_return_mean": float(gt_mean),
            "gt_return_std": float(gt_std),
            "reward_model_final_loss": float(rm_losses[-1]),
            "avg_trajectory_length": float(avg_len),
            "trajectories_reaching_goal": int(n_reached),
        })

    # --- Save results ---
    results_path = os.path.join(args.save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("RLHF Pipeline Complete")
    print(f"{'=' * 60}")
    print(f"\nResults summary:")
    for r in results:
        label = f"Round {r['round']} ({r['type']})"
        print(f"  {label:25s} GT return: {r['gt_return_mean']:7.1f} +/- {r['gt_return_std']:.1f}")
    print(f"\nResults saved to {results_path}")
    print(f"Models saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
