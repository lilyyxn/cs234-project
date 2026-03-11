"""Combined analysis: hiding incentive exists, but standard PPO cannot act on it.

Runs both experiments across multiple seeds and produces a clean 4-panel figure
with error bars saved to plots/combined_analysis.png.

Panel layout:
  Top-left:     GT return — proxy policy vs RLHF policy (correction works)
  Top-right:    Hiding incentive — proxy return drop when corrected
  Bottom-left:  Behavioral hiding score per RLHF round (~0 = cannot hide)
  Bottom-right: GT return at flag=0 vs flag=1 after RLHF (no behavioral gap)

Usage:
  python grid_world_env/plot_combined_analysis.py --fast
  python grid_world_env/plot_combined_analysis.py
  python grid_world_env/plot_combined_analysis.py --n-seeds 3
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from grid_world_env.train_rlhf import run_rlhf
from grid_world_env.train_behavioral_hiding import run_behavioral_hiding

PROXY_COLOR = "#e07b54"   # orange
RLHF_COLOR  = "#5b9bd5"   # blue
FLAG0_COLOR = "#e07b54"   # orange — unobserved
FLAG1_COLOR = "#5b9bd5"   # blue   — observed


# ---------------------------------------------------------------------------
# Run both experiments for one seed
# ---------------------------------------------------------------------------

def run_one_seed(seed, proxy_timesteps, n_rlhf_rounds, n_trajectories,
                 n_pairs, rm_epochs, rlhf_timesteps, eval_episodes, device):
    print(f"\n--- Seed {seed} ---")

    r = run_rlhf(
        proxy_timesteps=proxy_timesteps,
        n_rlhf_rounds=n_rlhf_rounds,
        n_trajectories_per_round=n_trajectories,
        n_pairs_per_round=n_pairs,
        reward_model_epochs=rm_epochs,
        rlhf_timesteps_per_round=rlhf_timesteps,
        eval_episodes=eval_episodes,
        seed=seed,
        device=device,
        verbose=False,
    )

    b = run_behavioral_hiding(
        proxy_timesteps=proxy_timesteps,
        n_rlhf_rounds=n_rlhf_rounds,
        n_trajectories_per_round=n_trajectories,
        n_pairs_per_round=n_pairs,
        reward_model_epochs=rm_epochs,
        rlhf_timesteps_per_round=rlhf_timesteps,
        eval_episodes=eval_episodes,
        seed=seed,
        device=device,
        verbose=False,
    )

    return {
        # Exp 1: hiding incentive
        "proxy_gt":          r["proxy_policy_gt_return"],
        "rlhf_gt":           r["rlhf_policy_gt_return"],
        "proxy_proxy":       r["proxy_policy_proxy_return"],
        "rlhf_proxy":        r["rlhf_policy_proxy_return"],
        "hiding_incentive":  r["hiding_incentive"],
        # Exp 2: behavioral hiding — hiding scores per round
        "hiding_scores":     b["hiding_score_per_round"],
        # GT return at flag=0 vs flag=1 after final round
        "final_gt_flag0":    b["round_metrics_flag0"][-1]["mean_gt_return"],
        "final_gt_flag1":    b["round_metrics_flag1"][-1]["mean_gt_return"],
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _bar_with_error(ax, x, mean, std, color, width=0.35, label=None):
    ax.bar(x, mean, width=width, color=color, edgecolor="white",
           linewidth=1.2, label=label)
    ax.errorbar(x, mean, yerr=std, fmt="none", color="black",
                capsize=5, capthick=1.5, linewidth=1.5)


def make_figure(all_results, n_rounds, out_path):
    # Aggregate across seeds
    def arr(key):
        return np.array([r[key] for r in all_results])

    proxy_gt   = arr("proxy_gt");    rlhf_gt   = arr("rlhf_gt")
    proxy_prx  = arr("proxy_proxy"); rlhf_prx  = arr("rlhf_proxy")
    hiding_inc = arr("hiding_incentive")
    # hiding scores: shape (n_seeds, n_rounds)
    hiding_scores = np.array([r["hiding_scores"] for r in all_results])
    fg0 = arr("final_gt_flag0");  fg1 = arr("final_gt_flag1")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        "Reward Hacking in RLHF\n"
        "The agent has a hiding incentive, but standard PPO cannot act on it",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ---- Panel 1: GT return before / after RLHF ----
    ax = axes[0]
    x = np.array([0, 1])
    _bar_with_error(ax, x[0], proxy_gt.mean(), proxy_gt.std(),
                    PROXY_COLOR, label="Proxy policy")
    _bar_with_error(ax, x[1], rlhf_gt.mean(),  rlhf_gt.std(),
                    RLHF_COLOR, label="RLHF policy")
    ax.set_xticks(x)
    ax.set_xticklabels(["Proxy policy\n(before RLHF)", "RLHF policy\n(after RLHF)"],
                        fontsize=9)
    ax.set_ylabel("Mean GT Return")
    ax.set_ylim(bottom=0)
    ax.set_title("1. RLHF correction works\nGT return improves after correction",
                 fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel 2: Proxy return before / after → hiding incentive ----
    ax = axes[1]
    _bar_with_error(ax, x[0], proxy_prx.mean(), proxy_prx.std(), PROXY_COLOR)
    _bar_with_error(ax, x[1], rlhf_prx.mean(),  rlhf_prx.std(),  RLHF_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(["Proxy policy\n(before RLHF)", "RLHF policy\n(after RLHF)"],
                        fontsize=9)
    ax.set_ylabel("Mean Proxy Return")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Annotate hiding incentive
    hi_mean = hiding_inc.mean()
    hi_std  = hiding_inc.std()
    y_top   = max(proxy_prx.mean(), rlhf_prx.mean())
    y_bot   = min(proxy_prx.mean(), rlhf_prx.mean())
    ax.annotate(
        f"Hiding incentive\n= {hi_mean:.1f} ± {hi_std:.1f}",
        xy=(0.5, (proxy_prx.mean() + rlhf_prx.mean()) / 2),
        xytext=(0.5, (proxy_prx.mean() + rlhf_prx.mean()) / 2),
        ha="center", va="center", fontsize=9, color="crimson", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=0.9),
    )
    ax.set_title("2. Correction creates a hiding incentive\nProxy return the agent loses when corrected",
                 fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel 3: Behavioral hiding score per round ----
    ax = axes[2]
    round_labels = [f"After\nRLHF {i+1}" for i in range(n_rounds)]
    x_rounds = np.arange(n_rounds)
    means = hiding_scores.mean(axis=0)
    stds  = hiding_scores.std(axis=0)
    colors = [PROXY_COLOR if m > 0 else RLHF_COLOR for m in means]
    for xi, m, s, c in zip(x_rounds, means, stds, colors):
        _bar_with_error(ax, xi, m, s, c, width=0.5)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    yabs = max(abs(means).max() * 1.5 + stds.max(), 1.0)
    ax.set_ylim(-yabs, yabs)
    ax.axhspan(-0.5, 0.5, alpha=0.1, color="green", label="Near-zero band (±0.5)")
    ax.set_xticks(x_rounds)
    ax.set_xticklabels(round_labels, fontsize=9)
    ax.set_ylabel("Hiding Score\n(flag=0 proxy) − (flag=1 proxy)")
    ax.set_title("3. Behavioral hiding score ≈ 0\nAgent cannot condition on eval flag",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel 4: GT return flag=0 vs flag=1 after final RLHF round ----
    ax = axes[3]
    x2 = np.array([0, 1])
    _bar_with_error(ax, x2[0], fg0.mean(), fg0.std(), FLAG0_COLOR,
                    label="flag=0 (unobserved)")
    _bar_with_error(ax, x2[1], fg1.mean(), fg1.std(), FLAG1_COLOR,
                    label="flag=1 (observed)")
    ax.set_xticks(x2)
    ax.set_xticklabels(["flag=0\n(unobserved)", "flag=1\n(observed)"], fontsize=9)
    ax.set_ylabel("Mean GT Return")
    ax.set_ylim(bottom=0)
    ax.set_title("4. No behavioral gap when observed\nGT return identical at flag=0 and flag=1",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Match the parameters that produced clean results in plot_rlhf.py
    parser.add_argument("--proxy-timesteps", type=int, default=300_000)
    parser.add_argument("--n-rlhf-rounds",   type=int, default=2)
    parser.add_argument("--n-trajectories",  type=int, default=150)
    parser.add_argument("--n-pairs",         type=int, default=300)
    parser.add_argument("--rm-epochs",       type=int, default=100)
    parser.add_argument("--rlhf-timesteps",  type=int, default=150_000)
    parser.add_argument("--eval-episodes",   type=int, default=100)
    parser.add_argument("--n-seeds",         type=int, default=3,
                        help="Number of seeds to run for error bars")
    parser.add_argument("--out-dir",         type=str, default="plots")
    parser.add_argument("--fast", action="store_true",
                        help="Tiny budget for quick verification")
    args = parser.parse_args()

    if args.fast:
        args.proxy_timesteps = 50_000
        args.n_rlhf_rounds   = 1
        args.n_trajectories  = 30
        args.n_pairs         = 60
        args.rm_epochs       = 10
        args.rlhf_timesteps  = 50_000
        args.eval_episodes   = 10
        args.n_seeds         = 2

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running combined analysis | device={device} | {args.n_seeds} seeds")

    all_results = []
    for seed in range(args.n_seeds):
        result = run_one_seed(
            seed=seed,
            proxy_timesteps=args.proxy_timesteps,
            n_rlhf_rounds=args.n_rlhf_rounds,
            n_trajectories=args.n_trajectories,
            n_pairs=args.n_pairs,
            rm_epochs=args.rm_epochs,
            rlhf_timesteps=args.rlhf_timesteps,
            eval_episodes=args.eval_episodes,
            device=device,
        )
        all_results.append(result)
        print(f"  Seed {seed}: hiding_incentive={result['hiding_incentive']:.2f}  "
              f"final_hiding_score={result['hiding_scores'][-1]:.3f}")

    out_path = os.path.join(args.out_dir, "combined_analysis.png")
    make_figure(all_results, args.n_rlhf_rounds, out_path)

    # Print summary
    hi = np.array([r["hiding_incentive"] for r in all_results])
    hs = np.array([r["hiding_scores"][-1] for r in all_results])
    print(f"\nSummary across {args.n_seeds} seeds:")
    print(f"  Hiding incentive:       {hi.mean():.2f} ± {hi.std():.2f}")
    print(f"  Final hiding score:     {hs.mean():.3f} ± {hs.std():.3f}")
    print(f"\nPlot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
