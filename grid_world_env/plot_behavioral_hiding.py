"""Plot results from the honest behavioral hiding experiment.

Runs run_behavioral_hiding() and saves four figures to plots/:
  1. hiding_score_over_rounds.png
  2. gt_return_by_flag.png
  3. proxy_return_by_flag.png
  4. behavioral_hiding_dashboard.png

Usage:
  python grid_world_env/plot_behavioral_hiding.py --fast
  python grid_world_env/plot_behavioral_hiding.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from grid_world_env.train_behavioral_hiding import run_behavioral_hiding

FLAG0_COLOR = "#e07b54"   # orange — flag=0 (unobserved)
FLAG1_COLOR = "#5b9bd5"   # blue   — flag=1 (observed)


def _stage_labels(n_rounds):
    labels = ["Proxy\n(pre-RLHF)"]
    for i in range(1, n_rounds + 1):
        labels.append(f"After\nRLHF {i}")
    return labels


def _build_series(results):
    pf0 = results["proxy_flag0"]
    pf1 = results["proxy_flag1"]
    rf0 = results["round_metrics_flag0"]
    rf1 = results["round_metrics_flag1"]
    gt0   = [pf0["mean_gt_return"]]    + [r["mean_gt_return"]    for r in rf0]
    gt1   = [pf1["mean_gt_return"]]    + [r["mean_gt_return"]    for r in rf1]
    pr0   = [pf0["mean_proxy_return"]] + [r["mean_proxy_return"] for r in rf0]
    pr1   = [pf1["mean_proxy_return"]] + [r["mean_proxy_return"] for r in rf1]
    hiding = ([pf0["mean_proxy_return"] - pf1["mean_proxy_return"]]
              + results["hiding_score_per_round"])
    return gt0, gt1, pr0, pr1, hiding


def plot_hiding_score(ax, stages, hiding_scores):
    colors = [FLAG0_COLOR if s > 0 else FLAG1_COLOR for s in hiding_scores]
    bars = ax.bar(stages, hiding_scores, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("Behavioral Hiding Score\n"
                 "(proxy_return[flag=0] − proxy_return[flag=1])\n"
                 "~0 = no hiding; +ve = agent exploits more when unobserved",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Hiding Score")
    for bar, val in zip(bars, hiding_scores):
        offset = max(abs(val) * 0.05, 0.1)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (offset if val >= 0 else -offset * 3),
                f"{val:.2f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric_by_flag(ax, stages, series0, series1, ylabel, title):
    x = np.arange(len(stages))
    w = 0.3
    ax.bar(x - w / 2, series0, width=w, color=FLAG0_COLOR,
           label="flag=0 (unobserved)", edgecolor="white")
    ax.bar(x + w / 2, series1, width=w, color=FLAG1_COLOR,
           label="flag=1 (observed)", edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-timesteps",  type=int, default=300_000)
    parser.add_argument("--n-rlhf-rounds",    type=int, default=2)
    parser.add_argument("--n-trajectories",   type=int, default=150)
    parser.add_argument("--n-pairs",          type=int, default=300)
    parser.add_argument("--rm-epochs",        type=int, default=100)
    parser.add_argument("--rlhf-timesteps",   type=int, default=150_000)
    parser.add_argument("--eval-episodes",    type=int, default=100)
    parser.add_argument("--seed",             type=int, default=0)
    parser.add_argument("--out-dir",          type=str, default="plots")
    parser.add_argument("--fast", action="store_true",
                        help="Tiny budget for quick smoke-test")
    args = parser.parse_args()

    if args.fast:
        args.proxy_timesteps  = 50_000
        args.n_rlhf_rounds    = 1
        args.n_trajectories   = 50
        args.n_pairs          = 100
        args.rm_epochs        = 20
        args.rlhf_timesteps   = 50_000
        args.eval_episodes    = 20

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running honest behavioral hiding experiment | device={device}")

    results = run_behavioral_hiding(
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

    n_rounds = args.n_rlhf_rounds
    stages = _stage_labels(n_rounds)
    gt0, gt1, pr0, pr1, hiding = _build_series(results)

    # Fig 1: hiding score
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_hiding_score(ax, stages, hiding)
    fig.tight_layout()
    p = os.path.join(args.out_dir, "hiding_score_over_rounds.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")

    # Fig 2: GT return by flag
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_metric_by_flag(ax, stages, gt0, gt1, "Mean GT Return",
                        "Ground-Truth Return: flag=0 vs flag=1")
    fig.tight_layout()
    p = os.path.join(args.out_dir, "gt_return_by_flag.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")

    # Fig 3: proxy return by flag
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_metric_by_flag(ax, stages, pr0, pr1, "Mean Proxy Return",
                        "Proxy Return: flag=0 vs flag=1\n"
                        "(gap near zero = no behavioral hiding with standard PPO)")
    fig.tight_layout()
    p = os.path.join(args.out_dir, "proxy_return_by_flag.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")

    # Fig 4: 2x2 dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Behavioral Hiding Experiment (Honest — Standard PPO)",
                 fontsize=14, fontweight="bold", y=1.01)
    plot_hiding_score(axes[0, 0], stages, hiding)
    plot_metric_by_flag(axes[0, 1], stages, gt0, gt1, "GT Return",
                        "Ground-Truth Return")
    plot_metric_by_flag(axes[1, 0], stages, pr0, pr1, "Proxy Return",
                        "Proxy Return (gap = hiding score)")
    pf0_both = ([results["proxy_flag0"]["pct_both_complete"]]
                + [r["pct_both_complete"] for r in results["round_metrics_flag0"]])
    pf1_both = ([results["proxy_flag1"]["pct_both_complete"]]
                + [r["pct_both_complete"] for r in results["round_metrics_flag1"]])
    plot_metric_by_flag(axes[1, 1], stages,
                        [v * 100 for v in pf0_both],
                        [v * 100 for v in pf1_both],
                        "Both-Phase Completion (%)", "Task Completion Rate")
    fig.tight_layout()
    p = os.path.join(args.out_dir, "behavioral_hiding_dashboard.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")

    # Numeric summary
    print("\n=== Behavioral Hiding Summary ===")
    print(f"{'Stage':<20} {'Hiding Score':>13} {'GT(0)':>7} {'GT(1)':>7} "
          f"{'Proxy(0)':>10} {'Proxy(1)':>10}")
    print("-" * 72)
    for stage, hs, g0, g1, p0, p1 in zip(stages, hiding, gt0, gt1, pr0, pr1):
        label = stage.replace("\n", " ")
        print(f"{label:<20} {hs:>13.3f} {g0:>7.3f} {g1:>7.3f} "
              f"{p0:>10.3f} {p1:>10.3f}")
    fhs = results["final_hiding_score"]
    print(f"\nFinal behavioral hiding score: {fhs:.3f}")
    print("  ~0 → standard PPO cannot learn to hide (expected honest result)")
    print("  +ve → unexpected hiding emerged (would be a novel finding)")
    print(f"\nPlots saved to: {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
