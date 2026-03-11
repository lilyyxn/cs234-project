"""Plot results from the parametric RLHF experiment.

Runs run_parametric_rlhf() and produces two figures saved to plots/:

  1. learned_reward_params.png
       Bar chart comparing original proxy reward parameters vs. learned
       parameters after each RLHF round. Directly shows what the preference
       labeler communicated to the reward model.

  2. parametric_rlhf_gt_return.png
       GT return across stages: proxy baseline → after each parametric RLHF round.
       Compared against the neural RLHF result from run_rlhf().

Usage:
  python grid_world_env/plot_parametric_rlhf.py
  python grid_world_env/plot_parametric_rlhf.py --fast
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from grid_world_env.train_parametric_rlhf import run_parametric_rlhf
from grid_world_env.train_rlhf import run_rlhf

PROXY_COLOR  = "#e07b54"   # orange
RLHF_COLOR   = "#5b9bd5"   # blue
NEURAL_COLOR = "#70ad47"   # green
ORIG_COLOR   = "#aaaaaa"   # grey  — original proxy params


def plot_learned_params(results, out_path):
    """Bar chart: original proxy params vs learned params after each round."""
    original = {
        "r_0_step": 0.0, "r_0_terminal": 1.0,
        "r_1_step": -2.0, "r_1_terminal": 0.0,
    }
    param_names  = ["r_0_step", "r_0_terminal", "r_1_step", "r_1_terminal"]
    display_names = ["r₀ step", "r₀ terminal", "r₁ step", "r₁ terminal"]
    n_rounds = len(results["learned_params_per_round"])

    # One group of bars per parameter, one bar per stage (original + each round)
    n_groups = len(param_names)
    n_bars   = 1 + n_rounds   # original + rounds
    x = np.arange(n_groups)
    width = 0.8 / n_bars

    colors = [ORIG_COLOR] + [RLHF_COLOR] * n_rounds
    labels = ["Original proxy"] + [f"After round {i+1}" for i in range(n_rounds)]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (color, label) in enumerate(zip(colors, labels)):
        if i == 0:
            vals = [original[p] for p in param_names]
        else:
            vals = [results["learned_params_per_round"][i - 1][p]
                    for p in param_names]
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      color=color, label=label, edgecolor="white")
        for bar, v in zip(bars, vals):
            ha = "center"
            va = "bottom" if v >= 0 else "top"
            yoff = 0.05 if v >= 0 else -0.05
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + yoff,
                    f"{v:.2f}", ha=ha, va=va, fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylabel("Parameter value")
    ax.set_title(
        "Learned Reward Parameters vs Original Proxy\n"
        "Preference learning maps what the proxy got wrong",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_gt_comparison(param_results, neural_results, out_path):
    """GT return: proxy baseline vs parametric RLHF vs neural RLHF."""
    n_rounds = len(param_results["round_metrics"])
    stages   = ["Proxy\n(pre-RLHF)"] + [f"After\nRLHF {i+1}" for i in range(n_rounds)]

    param_gt  = ([param_results["proxy_policy_gt_return"]]
                 + [r["mean_gt_return"] for r in param_results["round_metrics"]])
    neural_gt = ([neural_results["proxy_policy_gt_return"]]
                 + [r["mean_gt_return"] for r in neural_results["round_metrics"]])

    x = np.arange(len(stages))
    w = 0.3

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_p = ax.bar(x - w / 2, param_gt,  width=w, color=RLHF_COLOR,
                    label="Parametric RLHF", edgecolor="white")
    bars_n = ax.bar(x + w / 2, neural_gt, width=w, color=NEURAL_COLOR,
                    label="Neural RLHF",     edgecolor="white")

    for bars, vals in [(bars_p, param_gt), (bars_n, neural_gt)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Mean GT Return")
    ax.set_ylim(bottom=0)
    ax.set_title(
        "GT Return: Parametric vs Neural RLHF\n"
        "Both start from the same proxy-trained baseline",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-timesteps", type=int, default=300_000)
    parser.add_argument("--n-rlhf-rounds",   type=int, default=2)
    parser.add_argument("--n-trajectories",  type=int, default=150)
    parser.add_argument("--n-pairs",         type=int, default=300)
    parser.add_argument("--rm-epochs",       type=int, default=100)
    parser.add_argument("--rlhf-timesteps",  type=int, default=150_000)
    parser.add_argument("--eval-episodes",   type=int, default=100)
    parser.add_argument("--seed",            type=int, default=0)
    parser.add_argument("--out-dir",         type=str, default="plots")
    parser.add_argument("--fast", action="store_true",
                        help="Tiny budget for quick smoke-test")
    args = parser.parse_args()

    if args.fast:
        args.proxy_timesteps = 50_000
        args.n_rlhf_rounds   = 1
        args.n_trajectories  = 50
        args.n_pairs         = 100
        args.rm_epochs       = 20
        args.rlhf_timesteps  = 50_000
        args.eval_episodes   = 20

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running parametric RLHF experiment | device={device}")

    param_results = run_parametric_rlhf(
        proxy_timesteps=args.proxy_timesteps,
        n_rlhf_rounds=args.n_rlhf_rounds,
        n_trajectories_per_round=args.n_trajectories,
        n_pairs_per_round=args.n_pairs,
        reward_model_epochs=args.rm_epochs,
        rlhf_timesteps_per_round=args.rlhf_timesteps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=device,
    )

    print("\nRunning neural RLHF for comparison...")
    neural_results = run_rlhf(
        proxy_timesteps=args.proxy_timesteps,
        n_rlhf_rounds=args.n_rlhf_rounds,
        n_trajectories_per_round=args.n_trajectories,
        n_pairs_per_round=args.n_pairs,
        reward_model_epochs=args.rm_epochs,
        rlhf_timesteps_per_round=args.rlhf_timesteps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=device,
        verbose=False,
    )

    plot_learned_params(
        param_results,
        os.path.join(args.out_dir, "learned_reward_params.png"),
    )
    plot_gt_comparison(
        param_results, neural_results,
        os.path.join(args.out_dir, "parametric_rlhf_gt_return.png"),
    )

    # Numeric summary
    final = param_results["final_learned_params"]
    print("\n=== Parametric RLHF Summary ===")
    print(f"  Proxy GT return:        {param_results['proxy_policy_gt_return']:.3f}")
    print(f"  Parametric RLHF GT:     {param_results['rlhf_policy_gt_return']:.3f}")
    print(f"  Neural RLHF GT:         {neural_results['rlhf_policy_gt_return']:.3f}")
    print(f"\n  Final learned parameters:")
    print(f"    r_0_step     = {final['r_0_step']:.4f}  (original:  0)")
    print(f"    r_0_terminal = {final['r_0_terminal']:.4f}  (original:  1)")
    print(f"    r_1_step     = {final['r_1_step']:.4f}  (original: -2)")
    print(f"    r_1_terminal = {final['r_1_terminal']:.4f}  (original:  0)")
    print(f"\nPlots saved to: {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
