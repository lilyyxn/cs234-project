"""
Plotting utilities for RLHF experiments.

Generates:
1. GT return across RLHF rounds (with error bars across seeds)
2. Trajectory visualizations (wandering vs goal-directed)
3. Reward model inspection (what does G predict across the grid?)
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from grid_world_env.reward_model import RewardModel
from grid_world_env.train_rlhf import make_proxy_env, collect_trajectories
from grid_world_env.simulated_teacher import ground_truth_return
from stable_baselines3 import PPO


def plot_gt_returns_across_rounds(all_results, save_path=None):
    """Bar/line chart of GT return across RLHF rounds with error bars across seeds."""
    seeds = sorted(all_results.keys(), key=lambda x: int(x))
    n_rounds = len(all_results[seeds[0]])

    # Gather per-round stats across seeds
    round_means = []
    round_stds = []
    round_labels = []
    for r_idx in range(n_rounds):
        values = [all_results[s][r_idx]["gt_return_mean"] for s in seeds]
        round_means.append(np.mean(values))
        round_stds.append(np.std(values))
        r_data = all_results[seeds[0]][r_idx]
        label = f"Round {r_data['round']}\n({r_data['type']})"
        round_labels.append(label)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(n_rounds)
    colors = ["#d62728"] + ["#1f77b4"] * (n_rounds - 1)

    bars = ax.bar(x, round_means, yerr=round_stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)

    # Plot individual seed points
    for s in seeds:
        values = [all_results[s][r_idx]["gt_return_mean"] for r_idx in range(n_rounds)]
        ax.scatter(x, values, color="black", s=20, zorder=5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(round_labels)
    ax.set_ylabel("Ground-Truth Return", fontsize=12)
    ax.set_title("RLHF Correction of Proxy Reward Misalignment", fontsize=13)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # Add optimal baseline (approx: avg Manhattan distance ~4, so ~96)
    ax.axhline(y=96, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(n_rounds - 0.5, 96 + 2, "≈ optimal", color="green", fontsize=9,
            ha="right", va="bottom")

    ax.set_ylim(min(round_means) - 30, 110)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_trajectory_comparison(proxy_model_path, rlhf_model_path, save_path=None,
                                n_episodes=3, seed=42):
    """Side-by-side trajectory visualizations: proxy policy vs RLHF policy."""
    proxy_model = PPO.load(proxy_model_path)
    rlhf_model = PPO.load(rlhf_model_path)

    fig, axes = plt.subplots(2, n_episodes, figsize=(4 * n_episodes, 8))
    if n_episodes == 1:
        axes = axes.reshape(2, 1)

    for col, (model, label) in enumerate([(proxy_model, "Proxy Policy (+1/step)"),
                                            (rlhf_model, "RLHF-Corrected Policy")]):
        for ep in range(n_episodes):
            ax = axes[col, ep]
            env = make_proxy_env()
            obs, _ = env.reset(seed=seed + ep)

            # Extract initial positions
            agent_start = obs[:2].copy()  # relative position
            positions = [np.array([0.0, 0.0])]  # start at origin (agent frame)
            done = False
            step = 0
            reached_goal = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                next_obs, _, terminated, truncated, _ = env.step(action)

                # Track movement in relative frame
                # Each step changes agent position; we track cumulative displacement
                direction_map = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}
                new_pos = positions[-1] + np.array(direction_map[action])
                positions.append(new_pos)

                if next_obs[0] == 0.0 and next_obs[1] == 0.0:
                    reached_goal = True

                obs = next_obs
                done = terminated or truncated
                step += 1

            env.close()
            positions = np.array(positions)

            # Plot on grid
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Draw path
            ax.plot(positions[:, 0], positions[:, 1], "b-", alpha=0.4, linewidth=1)
            ax.plot(positions[0, 0], positions[0, 1], "go", markersize=10,
                    label="Start", zorder=5)
            ax.plot(positions[-1, 0], positions[-1, 1], "rs" if reached_goal else "rx",
                    markersize=10, label="End", zorder=5)

            # Target is at the initial relative position from obs
            target_rel = np.array([env.unwrapped._target_location[0] - env.unwrapped._agent_location[0],
                                    env.unwrapped._target_location[1] - env.unwrapped._agent_location[1]])

            status = f"{'GOAL' if reached_goal else 'TIMEOUT'} ({step} steps)"
            ax.set_title(f"{status}", fontsize=10)

            if ep == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")

    plt.suptitle("Agent Trajectories: Proxy vs RLHF", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_reward_model_heatmap(reward_model_path, save_path=None):
    """Visualize what G(s,a,s') predicts for each action at each relative position.

    Shows a heatmap of the reward correction across the grid for each action direction.
    """
    rm = RewardModel()
    rm.load_state_dict(torch.load(reward_model_path, weights_only=True))
    rm.eval()

    action_names = ["Right (+x)", "Up (+y)", "Left (-x)", "Down (-y)"]
    direction_map = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for action_idx, (ax, name) in enumerate(zip(axes, action_names)):
        grid_size = 9  # -4 to +4 relative positions
        heatmap = np.zeros((grid_size, grid_size))

        for i, rel_x in enumerate(range(-4, 5)):
            for j, rel_y in enumerate(range(-4, 5)):
                obs = np.array([rel_x, rel_y, 0.0, 0.5], dtype=np.float32)
                direction = direction_map[action_idx]
                next_rel_x = rel_x - direction[0]  # agent moves, so relative position changes opposite
                next_rel_y = rel_y - direction[1]
                next_obs = np.array([next_rel_x, next_rel_y, 0.0, 0.5], dtype=np.float32)

                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs)
                    a_onehot = torch.zeros(4)
                    a_onehot[action_idx] = 1.0
                    next_obs_t = torch.FloatTensor(next_obs)
                    g = rm(obs_t, a_onehot, next_obs_t).item()
                heatmap[j, i] = g

        im = ax.imshow(heatmap, origin="lower", cmap="RdBu_r",
                        extent=[-4.5, 4.5, -4.5, 4.5],
                        vmin=-np.max(np.abs(heatmap)), vmax=np.max(np.abs(heatmap)))
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Relative X to target")
        if action_idx == 0:
            ax.set_ylabel("Relative Y to target")
        ax.plot(0, 0, "k*", markersize=15, label="Target")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Learned Reward Correction G(s, a, s') by Action",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_preference_signal_analysis(all_results, save_path=None):
    """Show how trajectory quality (goal-reaching rate) affects RLHF success."""
    seeds = sorted(all_results.keys(), key=lambda x: int(x))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: goal-reaching rate per round per seed
    for s in seeds:
        rounds = []
        goal_rates = []
        for r_data in all_results[s]:
            if "trajectories_reaching_goal" in r_data:
                rounds.append(r_data["round"])
                n_traj = r_data.get("avg_trajectory_length", 500)
                # The n_trajectories is in the args, default 500
                goal_rates.append(r_data["trajectories_reaching_goal"] / 500 * 100)
        if rounds:
            ax1.plot(rounds, goal_rates, "o-", alpha=0.6, label=f"Seed {s}")

    ax1.set_xlabel("RLHF Round", fontsize=11)
    ax1.set_ylabel("Trajectories Reaching Goal (%)", fontsize=11)
    ax1.set_title("Trajectory Quality per Round", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: reward model loss per round per seed
    for s in seeds:
        rounds = []
        losses = []
        for r_data in all_results[s]:
            if "reward_model_final_loss" in r_data:
                rounds.append(r_data["round"])
                losses.append(r_data["reward_model_final_loss"])
        if rounds:
            ax2.plot(rounds, losses, "o-", alpha=0.6, label=f"Seed {s}")

    ax2.axhline(y=np.log(2), color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(0.5, np.log(2) + 0.02, "random guessing (ln 2)", color="red",
             fontsize=9, ha="left")
    ax2.set_xlabel("RLHF Round", fontsize=11)
    ax2.set_ylabel("Reward Model Loss", fontsize=11)
    ax2.set_title("Reward Model Learning", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Preference Signal Quality", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot RLHF results")
    parser.add_argument("--results-dir", type=str, default="models/rlhf_multiseed",
                        help="Directory with multi-seed results")
    parser.add_argument("--output-dir", type=str, default="plots/rlhf",
                        help="Directory to save plots")
    parser.add_argument("--single-seed-dir", type=str, default=None,
                        help="Single seed directory for trajectory/heatmap plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load multi-seed results
    agg_path = os.path.join(args.results_dir, "all_results.json")
    if os.path.exists(agg_path):
        with open(agg_path) as f:
            all_results = json.load(f)

        plot_gt_returns_across_rounds(
            all_results,
            save_path=os.path.join(args.output_dir, "gt_returns_across_rounds.png"),
        )

        plot_preference_signal_analysis(
            all_results,
            save_path=os.path.join(args.output_dir, "preference_signal.png"),
        )
    else:
        print(f"No multi-seed results at {agg_path}. Run run_rlhf_seeds.py first.")

    # Single-seed plots (trajectories and heatmap)
    seed_dir = args.single_seed_dir
    if seed_dir is None:
        # Try to find a seed that succeeded
        for s in sorted(os.listdir(args.results_dir)):
            candidate = os.path.join(args.results_dir, s)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "results.json")):
                seed_dir = candidate
                break

    if seed_dir and os.path.isdir(seed_dir):
        proxy_path = os.path.join(seed_dir, "ppo_proxy")
        # Find the last RLHF round model
        rlhf_paths = sorted([f for f in os.listdir(seed_dir) if f.startswith("ppo_rlhf_round")])
        if rlhf_paths and os.path.exists(proxy_path + ".zip"):
            last_rlhf = os.path.join(seed_dir, rlhf_paths[-1].replace(".zip", ""))
            plot_trajectory_comparison(
                proxy_path, last_rlhf,
                save_path=os.path.join(args.output_dir, "trajectory_comparison.png"),
            )

        # Reward model heatmap
        rm_paths = sorted([f for f in os.listdir(seed_dir) if f.startswith("reward_model_round")])
        if rm_paths:
            last_rm = os.path.join(seed_dir, rm_paths[-1])
            plot_reward_model_heatmap(
                last_rm,
                save_path=os.path.join(args.output_dir, "reward_correction_heatmap.png"),
            )

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
