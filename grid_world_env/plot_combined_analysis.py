"""Combined analysis: hiding incentive exists, but standard PPO cannot act on it.

Runs both experiments and produces a single unified figure that tells the
complete story in two parts:

  Part 1 — Hiding Incentive (from run_rlhf):
    The proxy-trained agent exploits the proxy reward but completes 0% of tasks.
    After RLHF correction it achieves 100% task completion, but its proxy return
    drops sharply. That drop IS the hiding incentive — what the agent stands to
    lose if corrected.

  Part 2 — Cannot Hide (from run_behavioral_hiding):
    Even when given an eval_flag observation bit (1 = "being evaluated"), the
    agent shows no behavioral gap between flag=0 and flag=1. Hiding score ≈ 0.
    Standard PPO cannot solve the long-horizon credit assignment required to
    learn strategic behavioral hiding.

Saves to plots/combined_analysis.png

Usage:
  python grid_world_env/plot_combined_analysis.py --fast
  python grid_world_env/plot_combined_analysis.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from grid_world_env.train_rlhf import run_rlhf
from grid_world_env.train_behavioral_hiding import run_behavioral_hiding


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PROXY_COLOR  = "#e07b54"   # orange — proxy policy / unobserved
RLHF_COLOR   = "#5b9bd5"   # blue   — RLHF policy / observed
GOOD_COLOR   = "#70ad47"   # green  — GT / task completion
NEUTRAL      = "#888888"   # grey   — zero line / neutral

# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def panel_hiding_incentive(ax, proxy_proxy, rlhf_proxy, hiding_incentive):
    """Bar: proxy return of proxy policy vs RLHF policy — the incentive to hide."""
    labels = ["Proxy policy\n(before RLHF)", "RLHF policy\n(after RLHF)"]
    values = [proxy_proxy, rlhf_proxy]
    bars = ax.bar(labels, values, color=[PROXY_COLOR, RLHF_COLOR],
                  width=0.45, edgecolor="white", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    for bar, val in zip(bars, values):
        off = max(abs(val) * 0.04, 0.3)
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (off if val >= 0 else -off * 2),
                f"{val:.1f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=10, fontweight="bold")

    # Annotate the gap
    ax.annotate(
        f"Hiding incentive\n= {hiding_incentive:.1f}",
        xy=(1, rlhf_proxy), xytext=(0, proxy_proxy),
        arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.8),
        color="crimson", fontsize=9, ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="crimson", alpha=0.85),
    )
    ax.set_title("Part 1: Hiding Incentive\n"
                 "Proxy return the agent LOSES when corrected",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Proxy Return")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_correction_works(ax, stages_rlhf, gt_series, proxy_series):
    """Line: GT vs proxy return across RLHF stages — correction is beneficial."""
    x = range(len(stages_rlhf))
    ax.plot(x, gt_series,    marker="o", color=GOOD_COLOR,   linewidth=2,
            label="GT return (task completion)", markersize=7)
    ax.plot(x, proxy_series, marker="s", color=PROXY_COLOR,  linewidth=2,
            linestyle="--", label="Proxy return (exploitation)", markersize=7)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.replace("\n", " ") for s in stages_rlhf], fontsize=8)
    ax.set_title("RLHF Correction Effectiveness\n"
                 "GT return ↑ while proxy return ↓",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Return")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_behavioral_gap(ax, stages_bh, proxy_flag0, proxy_flag1):
    """Grouped bars: proxy return at flag=0 vs flag=1 per stage."""
    x = np.arange(len(stages_bh))
    w = 0.3
    ax.bar(x - w / 2, proxy_flag0, width=w, color=PROXY_COLOR,
           label="flag=0 (unobserved)", edgecolor="white")
    ax.bar(x + w / 2, proxy_flag1, width=w, color=RLHF_COLOR,
           label="flag=1 (observed)", edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("\n", " ") for s in stages_bh], fontsize=8)
    ax.set_title("Part 2: Behavioral Gap\n"
                 "Proxy return unobserved vs observed",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Proxy Return")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_hiding_score(ax, stages_bh, hiding_scores):
    """Bar: behavioral hiding score per stage — expected ~0."""
    colors = [PROXY_COLOR if s > 0 else RLHF_COLOR for s in hiding_scores]
    bars = ax.bar(stages_bh, hiding_scores, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

    for bar, val in zip(bars, hiding_scores):
        off = max(abs(val) * 0.05, 0.1)
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (off if val >= 0 else -off * 3),
                f"{val:.2f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=9)

    # Add "~0" annotation band
    ymax = max(abs(min(hiding_scores)), abs(max(hiding_scores))) * 1.5 + 0.5
    ax.set_ylim(-ymax, ymax)
    ax.axhspan(-0.5, 0.5, alpha=0.08, color=GOOD_COLOR,
               label="Near-zero band (±0.5)")
    ax.set_title("Behavioral Hiding Score\n"
                 "(flag=0 proxy return) − (flag=1 proxy return)\n"
                 "~0 = standard PPO cannot learn to hide",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Hiding Score")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combined hiding incentive + behavioral hiding analysis"
    )
    parser.add_argument("--proxy-timesteps",  type=int, default=200_000)
    parser.add_argument("--n-rlhf-rounds",    type=int, default=2)
    parser.add_argument("--n-trajectories",   type=int, default=100)
    parser.add_argument("--n-pairs",          type=int, default=200)
    parser.add_argument("--rm-epochs",        type=int, default=50)
    parser.add_argument("--rlhf-timesteps",   type=int, default=100_000)
    parser.add_argument("--eval-episodes",    type=int, default=50)
    parser.add_argument("--seed",             type=int, default=0)
    parser.add_argument("--out-dir",          type=str, default="plots")
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
        args.eval_episodes   = 20

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Experiment 1: Hiding incentive ----
    print("=" * 60)
    print("Running Experiment 1: Hiding Incentive (run_rlhf)")
    print("=" * 60)
    rlhf_results = run_rlhf(
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

    # ---- Experiment 2: Behavioral hiding ----
    print("\n" + "=" * 60)
    print("Running Experiment 2: Behavioral Hiding (run_behavioral_hiding)")
    print("=" * 60)
    bh_results = run_behavioral_hiding(
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

    # ---- Build series ----
    n_rounds = args.n_rlhf_rounds

    # RLHF experiment series
    rlhf_stages = (["Proxy\n(pre-RLHF)"]
                   + [f"After\nRLHF {i}" for i in range(1, n_rounds + 1)])
    rlhf_gt    = ([rlhf_results["proxy_policy_gt_return"]]
                  + [r["mean_gt_return"]    for r in rlhf_results["round_metrics"]])
    rlhf_proxy = ([rlhf_results["proxy_policy_proxy_return"]]
                  + [r["mean_proxy_return"] for r in rlhf_results["round_metrics"]])

    # Behavioral hiding series
    bh_stages  = (["Proxy\n(pre-RLHF)"]
                  + [f"After\nRLHF {i}" for i in range(1, n_rounds + 1)])
    bh_pr0     = ([bh_results["proxy_flag0"]["mean_proxy_return"]]
                  + [r["mean_proxy_return"] for r in bh_results["round_metrics_flag0"]])
    bh_pr1     = ([bh_results["proxy_flag1"]["mean_proxy_return"]]
                  + [r["mean_proxy_return"] for r in bh_results["round_metrics_flag1"]])
    bh_hiding  = ([bh_results["proxy_flag0"]["mean_proxy_return"]
                   - bh_results["proxy_flag1"]["mean_proxy_return"]]
                  + bh_results["hiding_score_per_round"])

    hiding_incentive = rlhf_results["hiding_incentive"]
    final_hiding_score = bh_results["final_hiding_score"]

    # ---- Figure ----
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.38)

    ax_incentive  = fig.add_subplot(gs[0, 0])
    ax_correction = fig.add_subplot(gs[0, 1])
    ax_gap        = fig.add_subplot(gs[1, 0])
    ax_score      = fig.add_subplot(gs[1, 1])

    panel_hiding_incentive(ax_incentive,
                           rlhf_results["proxy_policy_proxy_return"],
                           rlhf_results["rlhf_policy_proxy_return"],
                           hiding_incentive)
    panel_correction_works(ax_correction, rlhf_stages, rlhf_gt, rlhf_proxy)
    panel_behavioral_gap(ax_gap, bh_stages, bh_pr0, bh_pr1)
    panel_hiding_score(ax_score, bh_stages, bh_hiding)

    # Column headers
    fig.text(0.28, 0.97,
             "Part 1 — The agent HAS an incentive to hide",
             ha="center", fontsize=12, fontweight="bold", color="#c0392b")
    fig.text(0.75, 0.97,
             "Part 2 — But CANNOT act on it (standard PPO)",
             ha="center", fontsize=12, fontweight="bold", color="#2980b9")

    # Main title
    fig.suptitle(
        f"Reward Hacking in RLHF: Hiding Incentive = {hiding_incentive:.1f},  "
        f"Behavioral Hiding Score = {final_hiding_score:.2f}",
        fontsize=13, y=1.03, fontweight="bold",
    )

    out_path = os.path.join(args.out_dir, "combined_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("COMBINED ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nExperiment 1 — Hiding Incentive:")
    print(f"  Proxy policy:  GT={rlhf_results['proxy_policy_gt_return']:.3f}  "
          f"proxy={rlhf_results['proxy_policy_proxy_return']:.3f}  "
          f"phase0={rlhf_results['proxy_pct_phase0']:.1%}  "
          f"both={rlhf_results['proxy_pct_both']:.1%}")
    print(f"  RLHF policy:   GT={rlhf_results['rlhf_policy_gt_return']:.3f}  "
          f"proxy={rlhf_results['rlhf_policy_proxy_return']:.3f}  "
          f"phase0={rlhf_results['rlhf_pct_phase0']:.1%}  "
          f"both={rlhf_results['rlhf_pct_both']:.1%}")
    print(f"  Hiding incentive:  {hiding_incentive:.3f}  "
          f"(proxy return lost when RLHF corrects the agent)")

    print(f"\nExperiment 2 — Behavioral Hiding (honest, standard PPO):")
    for i, (hs, m0, m1) in enumerate(zip(
            bh_hiding[1:],
            bh_results["round_metrics_flag0"],
            bh_results["round_metrics_flag1"])):
        print(f"  Round {i+1}:  flag=0 GT={m0['mean_gt_return']:.2f} proxy={m0['mean_proxy_return']:.2f}  |  "
              f"flag=1 GT={m1['mean_gt_return']:.2f} proxy={m1['mean_proxy_return']:.2f}  |  "
              f"hiding_score={hs:.3f}")
    print(f"  Final hiding score: {final_hiding_score:.3f}  (~0 = no behavioral hiding)")

    print(f"\nConclusion:")
    print(f"  The agent has a structural hiding incentive of {hiding_incentive:.1f}")
    print(f"  (it loses this much proxy return when corrected).")
    print(f"  But with standard PPO, behavioral hiding score = {final_hiding_score:.2f} ≈ 0.")
    print(f"  Standard PPO cannot learn to conditionally suppress its exploit")
    print(f"  based on whether it is being observed.")
    print(f"\nPlot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
