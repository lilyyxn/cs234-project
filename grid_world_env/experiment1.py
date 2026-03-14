"""Experiment 1: Multi-round parametric RLHF with behavioral avoidance of oversight.

Pipeline summary:
  1. Train a proxy policy using potential-shaping reward (encourages looping near Phase-0 target).
  2. For each RLHF round:
     a. Collect trajectories from the current policy.
     b. Split into phase1_trajs (entered Phase 1) and phase0_only_trajs (never left Phase 0).
     c. Compute the hiding incentive: mean proxy return for phase0-only episodes (gt_return=0
        for those, so proxy_return - gt_return = proxy_return).
     d. If enough phase1_trajs exist, generate preference pairs from them and train the
        parametric reward model on those pairs only.
     e. Compute mean_phase1_rlhf_return using the updated reward parameters.
     f. Retrain a new policy from scratch on an env built from the updated parameters.
  3. Plot three panels: hiding incentive, expected Phase-1 RLHF return, Phase-0 completion rate.

Run:
  python grid_world_env/experiment1.py
  python grid_world_env/experiment1.py --fast
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make the package importable when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import grid_world_env  # registers the gym env
from grid_world_env.train_ppo_scratch import make_env, train, ActorCritic
from grid_world_env.rlhf.preference_data import (
    collect_trajectory,
    generate_preference_pairs,
)
from grid_world_env.rlhf.parametric_reward_model import (
    ParametricRewardModel,
    train_parametric_reward_model,
)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_policy(policy, env, n_episodes=100, device="cpu"):
    """Evaluate policy on env for n_episodes, returning key metrics."""
    trajs = [collect_trajectory(policy, env, device=device) for _ in range(n_episodes)]
    gt_returns = [t.gt_return for t in trajs]
    proxy_returns = [t.proxy_return for t in trajs]
    return {
        "mean_gt_return": float(np.mean(gt_returns)),
        "mean_proxy_return": float(np.mean(proxy_returns)),
        "pct_phase0_complete": float(np.mean([r >= 1.0 for r in gt_returns])),
        "pct_both_complete": float(np.mean([r >= 2.0 for r in gt_returns])),
    }


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_experiment1(
    proxy_timesteps: int,
    n_rounds: int,
    n_trajectories: int,
    n_pairs: int,
    rm_epochs: int,
    rlhf_timesteps: int,
    eval_episodes: int,
    seed: int,
    device: str = "cpu",
):
    """Run one seed of Experiment 1.

    Returns:
        dict with keys:
          "proxy_metrics": evaluate_policy result for the initial proxy policy
          "rounds": list of per-round metric dicts
    """
    rng_seed = seed

    # ------------------------------------------------------------------
    # Step 1: Train proxy policy
    # ------------------------------------------------------------------
    print(f"[seed={seed}] Training proxy policy ({proxy_timesteps} steps)...")
    proxy_env = make_env(
        use_potential_shaping=True,
        reward_0_step=0,
        reward_0_terminal=1,
        reward_1_step=-2,
        reward_1_terminal=0,
    )
    current_policy = train(
        proxy_env,
        total_timesteps=proxy_timesteps,
        seed=rng_seed,
        device=device,
    )

    proxy_policy = current_policy  # fixed reference for data collection every round
    proxy_metrics = evaluate_policy(proxy_policy, proxy_env, n_episodes=eval_episodes, device=device)
    print(f"[seed={seed}] Proxy policy: {proxy_metrics}")

    # Initialise parametric reward model at proxy reward values (no Phase-1 terminal)
    reward_model = ParametricRewardModel(
        r_0_step=0.0,
        r_0_terminal=1.0,
        r_1_step=0.0,
        r_1_terminal=0.0,
    )

    per_round_results = []

    for rnd in range(n_rounds):
        print(f"\n[seed={seed}] === Round {rnd + 1}/{n_rounds} ===")

        # ------------------------------------------------------------------
        # Step 2a: Collect trajectories from proxy policy on proxy env
        # ------------------------------------------------------------------
        trajs = [
            collect_trajectory(proxy_policy, proxy_env, device=device)
            for _ in range(n_trajectories)
        ]

        # ------------------------------------------------------------------
        # Step 2b: Split by whether Phase 1 was entered
        # ------------------------------------------------------------------
        phase1_trajs = [t for t in trajs if np.any(t.observations[:, 2] == 1.0)]
        phase0_only_trajs = [t for t in trajs if not np.any(t.observations[:, 2] == 1.0)]

        phase0_completion_rate = len(phase1_trajs) / len(trajs)

        # ------------------------------------------------------------------
        # Step 2c: Hiding incentive (mean proxy return for phase0-only trajs)
        # ------------------------------------------------------------------
        if phase0_only_trajs:
            mean_hiding_incentive = float(
                np.mean([t.proxy_return - t.gt_return for t in phase0_only_trajs])
            )
        else:
            mean_hiding_incentive = 0.0

        # ------------------------------------------------------------------
        # Step 2d: Update reward model from Phase-1 trajectories only
        # ------------------------------------------------------------------
        if len(phase1_trajs) >= 2:
            pairs = generate_preference_pairs(phase1_trajs, n_pairs=n_pairs, seed=rng_seed + rnd)
            reward_model = train_parametric_reward_model(reward_model, pairs, n_epochs=rm_epochs)
            learned_params = reward_model.as_dict()
        else:
            learned_params = reward_model.as_dict()
            print(f"  [seed={seed}] Only {len(phase1_trajs)} phase1 traj(s); skipping reward model update.")

        learned_r1_terminal = learned_params["r_1_terminal"]
        print(f"  [seed={seed}] Learned params: {learned_params}")
        print(f"  [seed={seed}] Phase-0 completion rate: {phase0_completion_rate:.3f}")
        print(f"  [seed={seed}] Hiding incentive: {mean_hiding_incentive:.3f}")

        # ------------------------------------------------------------------
        # Step 2e: Mean Phase-1 RLHF return under updated params
        # ------------------------------------------------------------------
        r_1_step = learned_params["r_1_step"]
        r_1_terminal = learned_params["r_1_terminal"]

        if phase1_trajs:
            rlhf_returns = []
            for traj in phase1_trajs:
                phases = traj.observations[:, 2]
                n1 = float(np.sum(phases == 1.0))
                term1 = 1.0 if (traj.terminated and n1 > 0) else 0.0
                rlhf_returns.append(r_1_step * n1 + r_1_terminal * term1)
            mean_phase1_rlhf_return = float(np.mean(rlhf_returns))
        else:
            mean_phase1_rlhf_return = 0.0

        print(f"  [seed={seed}] Mean Phase-1 RLHF return: {mean_phase1_rlhf_return:.3f}")

        # ------------------------------------------------------------------
        # Step 2f: Build updated env and fine-tune policy
        # ------------------------------------------------------------------
        updated_env = make_env(
            use_potential_shaping=False,
            reward_0_step=learned_params["r_0_step"],
            reward_0_terminal=learned_params["r_0_terminal"],
            reward_1_step=learned_params["r_1_step"],
            reward_1_terminal=learned_params["r_1_terminal"],
        )
        current_policy = train(
            updated_env,
            total_timesteps=rlhf_timesteps,
            seed=rng_seed + rnd + 1,
            device=device,
        )

        # ------------------------------------------------------------------
        # Step 2h: Evaluate on proxy env
        # ------------------------------------------------------------------
        eval_metrics = evaluate_policy(current_policy, proxy_env, n_episodes=eval_episodes, device=device)
        print(f"  [seed={seed}] Post-round eval: {eval_metrics}")

        per_round_results.append({
            "round": rnd + 1,
            "phase0_completion_rate": phase0_completion_rate,
            "mean_hiding_incentive": mean_hiding_incentive,
            "learned_r1_terminal": learned_r1_terminal,
            "mean_phase1_rlhf_return": mean_phase1_rlhf_return,
            "eval": eval_metrics,
        })

    return {
        "proxy_metrics": proxy_metrics,
        "rounds": per_round_results,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_seed_results: list, n_rounds: int, save_path: str):
    """Average per-round metrics across seeds and plot three panels."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rounds = list(range(1, n_rounds + 1))

    def extract_metric(key):
        """Return (means, stds) arrays of length n_rounds across seeds."""
        per_seed = []
        for seed_result in all_seed_results:
            per_seed.append([r[key] for r in seed_result["rounds"]])
        arr = np.array(per_seed)  # (n_seeds, n_rounds)
        return arr.mean(axis=0), arr.std(axis=0)

    hiding_mean, hiding_std = extract_metric("mean_hiding_incentive")
    phase1_rlhf_mean, phase1_rlhf_std = extract_metric("mean_phase1_rlhf_return")
    comp_mean, comp_std = extract_metric("phase0_completion_rate")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Hiding incentive
    ax = axes[0]
    ax.plot(rounds, hiding_mean, marker="o", color="C0")
    ax.fill_between(rounds,
                    hiding_mean - hiding_std,
                    hiding_mean + hiding_std,
                    alpha=0.25, color="C0")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean Proxy Return (Phase-0-Only)")
    ax.set_title("Hiding Incentive (Phase-0-Only Episodes)")
    ax.set_xticks(rounds)

    # Panel 2: Expected Phase-1 return under learned reward
    ax = axes[1]
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.plot(rounds, phase1_rlhf_mean, marker="o", color="C1")
    ax.fill_between(rounds,
                    phase1_rlhf_mean - phase1_rlhf_std,
                    phase1_rlhf_mean + phase1_rlhf_std,
                    alpha=0.25, color="C1")
    ax.set_xlabel("Round")
    ax.set_ylabel("Expected RLHF Return")
    ax.set_title("Expected Phase 1 Return Under Learned Reward")
    ax.set_xticks(rounds)

    # Panel 3: Phase-0 completion rate
    ax = axes[2]
    ax.plot(rounds, comp_mean, marker="o", color="C2")
    ax.fill_between(rounds,
                    comp_mean - comp_std,
                    comp_mean + comp_std,
                    alpha=0.25, color="C2")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Round")
    ax.set_ylabel("Fraction of Trajectories")
    ax.set_title("Phase 0 Completion Rate")
    ax.set_xticks(rounds)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Multi-round parametric RLHF with behavioral avoidance of oversight."
    )
    parser.add_argument("--proxy-timesteps", type=int, default=300_000)
    parser.add_argument("--n-rounds", type=int, default=6)
    parser.add_argument("--n-trajectories", type=int, default=150)
    parser.add_argument("--n-pairs", type=int, default=300)
    parser.add_argument("--rm-epochs", type=int, default=100)
    parser.add_argument("--rlhf-timesteps", type=int, default=150_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Reduce budgets for quick testing.",
    )
    args = parser.parse_args()

    if args.fast:
        args.proxy_timesteps = 50_000
        args.n_trajectories = 30
        args.n_pairs = 60
        args.rm_epochs = 20
        args.rlhf_timesteps = 50_000
        args.eval_episodes = 20
        args.n_seeds = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: proxy_timesteps={args.proxy_timesteps}, n_rounds={args.n_rounds}, "
          f"n_trajectories={args.n_trajectories}, n_pairs={args.n_pairs}, "
          f"rm_epochs={args.rm_epochs}, rlhf_timesteps={args.rlhf_timesteps}, "
          f"eval_episodes={args.eval_episodes}, n_seeds={args.n_seeds}")

    all_seed_results = []
    for s in range(args.n_seeds):
        seed = args.seed + s
        print(f"\n{'='*60}")
        print(f"Seed {s + 1}/{args.n_seeds}  (seed={seed})")
        print(f"{'='*60}")
        result = run_experiment1(
            proxy_timesteps=args.proxy_timesteps,
            n_rounds=args.n_rounds,
            n_trajectories=args.n_trajectories,
            n_pairs=args.n_pairs,
            rm_epochs=args.rm_epochs,
            rlhf_timesteps=args.rlhf_timesteps,
            eval_episodes=args.eval_episodes,
            seed=seed,
            device=device,
        )
        all_seed_results.append(result)

    plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    save_path = os.path.join(plots_dir, "experiment1.png")
    plot_results(all_seed_results, n_rounds=args.n_rounds, save_path=save_path)


if __name__ == "__main__":
    main()
