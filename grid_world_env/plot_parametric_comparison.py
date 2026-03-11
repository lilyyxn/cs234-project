"""Parametric RLHF: 1-round vs 2-round comparison with error bars.

Runs run_parametric_rlhf() with n_rlhf_rounds=2 per seed and extracts both
after-round-1 and after-round-2 metrics from round_metrics — so the proxy
policy is only trained once per seed.

Key finding:
  1-round RLHF fails: proxy policy never completes Phase 0, so all preference
  pairs compare GT=0 vs GT=0, labels are random, and r_1_terminal stays near
  zero.  The agent trained on the unchanged parameters still avoids Phase 1.

  2-round RLHF succeeds (bootstrapping): Round 1 creates a weak positive
  signal for Phase 1 entry, causing the policy to complete Phase 0 in Round 2.
  Round 2 then has real Phase 1 completion data and learns r_1_terminal >> 0.

Figure saved to plots/parametric_comparison.png (2 panels):
  Panel 1: GT return — proxy baseline vs after round 1 vs after round 2
  Panel 2: Learned r_1_terminal — round 1 vs round 2
           (r_1_terminal ≈ 0 after round 1, large after round 2)

Usage:
  python grid_world_env/plot_parametric_comparison.py
  python grid_world_env/plot_parametric_comparison.py --fast
  python grid_world_env/plot_parametric_comparison.py --n-seeds 5
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from grid_world_env.train_parametric_rlhf import run_parametric_rlhf

PROXY_COLOR  = "#e07b54"   # orange — proxy baseline
ROUND1_COLOR = "#d94f4f"   # red    — 1-round (fails)
ROUND2_COLOR = "#5b9bd5"   # blue   — 2-round (succeeds)


def _bar_err(ax, x, mean, std, color, width=0.5, label=None):
    ax.bar(x, mean, width=width, color=color, edgecolor="white",
           linewidth=1.2, label=label)
    ax.errorbar(x, mean, yerr=std, fmt="none", color="black",
                capsize=5, capthick=1.5, linewidth=1.5)


def run_one_seed(seed, proxy_timesteps, n_trajectories, n_pairs,
                 rm_epochs, rlhf_timesteps, eval_episodes, device):
    print(f"\n--- Seed {seed} ---")
    r = run_parametric_rlhf(
        proxy_timesteps=proxy_timesteps,
        n_rlhf_rounds=2,
        n_trajectories_per_round=n_trajectories,
        n_pairs_per_round=n_pairs,
        reward_model_epochs=rm_epochs,
        rlhf_timesteps_per_round=rlhf_timesteps,
        eval_episodes=eval_episodes,
        seed=seed,
        device=device,
        verbose=False,
    )
    p1 = r["learned_params_per_round"][0]
    p2 = r["learned_params_per_round"][1]
    print(f"  Proxy  GT: {r['proxy_policy_gt_return']:.3f}")
    print(f"  Round1 GT: {r['round_metrics'][0]['mean_gt_return']:.3f}  "
          f"r_1_terminal={p1['r_1_terminal']:.3f}")
    print(f"  Round2 GT: {r['round_metrics'][1]['mean_gt_return']:.3f}  "
          f"r_1_terminal={p2['r_1_terminal']:.3f}")
    return {
        "proxy_gt":        r["proxy_policy_gt_return"],
        "round1_gt":       r["round_metrics"][0]["mean_gt_return"],
        "round2_gt":       r["round_metrics"][1]["mean_gt_return"],
        "round1_r1t":      p1["r_1_terminal"],
        "round2_r1t":      p2["r_1_terminal"],
        "round1_r1s":      p1["r_1_step"],
        "round2_r1s":      p2["r_1_step"],
    }


def make_figure(all_results, out_path):
    def arr(key):
        return np.array([r[key] for r in all_results])

    proxy_gt  = arr("proxy_gt")
    round1_gt = arr("round1_gt")
    round2_gt = arr("round2_gt")
    round1_r1t = arr("round1_r1t")
    round2_r1t = arr("round2_r1t")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Parametric RLHF: 1-Round Fails, 2-Round Succeeds via Bootstrapping\n"
        "Plugging learned reward parameters directly into the environment",
        fontsize=12, fontweight="bold", y=1.03,
    )

    # ---- Panel 1: GT return across stages ----
    ax = axes[0]
    x = np.array([0, 1, 2])
    _bar_err(ax, x[0], proxy_gt.mean(),  proxy_gt.std(),  PROXY_COLOR,
             label="Proxy baseline")
    _bar_err(ax, x[1], round1_gt.mean(), round1_gt.std(), ROUND1_COLOR,
             label="After round 1 (fails)")
    _bar_err(ax, x[2], round2_gt.mean(), round2_gt.std(), ROUND2_COLOR,
             label="After round 2 (bootstrapped)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Proxy\nbaseline", "1-round\nRLHF", "2-round\nRLHF"], fontsize=9
    )
    ax.set_ylabel("Mean GT Return")
    ax.set_ylim(bottom=0, top=2.2)
    ax.axhline(2.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.7,
               label="Max GT return (2.0)")
    ax.set_title(
        "GT Return: 1-Round RLHF Fails\n"
        "Single round has no Phase 1 signal",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel 2: Learned r_1_terminal ----
    ax = axes[1]
    x2 = np.array([0, 1])
    _bar_err(ax, x2[0], round1_r1t.mean(), round1_r1t.std(), ROUND1_COLOR,
             label="After round 1")
    _bar_err(ax, x2[1], round2_r1t.mean(), round2_r1t.std(), ROUND2_COLOR,
             label="After round 2")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="Proxy value (0)")
    ax.set_xticks(x2)
    ax.set_xticklabels(["After\nround 1", "After\nround 2"], fontsize=9)
    ax.set_ylabel("Learned r₁_terminal")
    ax.set_title(
        "Learned Phase 1 Terminal Reward\n"
        "Round 1 has no signal; round 2 corrects via bootstrapping",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-timesteps", type=int, default=300_000)
    parser.add_argument("--n-trajectories",  type=int, default=150)
    parser.add_argument("--n-pairs",         type=int, default=300)
    parser.add_argument("--rm-epochs",       type=int, default=100)
    parser.add_argument("--rlhf-timesteps",  type=int, default=150_000)
    parser.add_argument("--eval-episodes",   type=int, default=100)
    parser.add_argument("--n-seeds",         type=int, default=3)
    parser.add_argument("--out-dir",         type=str, default="plots")
    parser.add_argument("--fast", action="store_true",
                        help="Tiny budget for quick smoke-test")
    args = parser.parse_args()

    if args.fast:
        args.proxy_timesteps = 50_000
        args.n_trajectories  = 30
        args.n_pairs         = 60
        args.rm_epochs       = 20
        args.rlhf_timesteps  = 50_000
        args.eval_episodes   = 20
        args.n_seeds         = 2

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Parametric RLHF comparison | device={device} | {args.n_seeds} seeds")

    all_results = []
    for seed in range(args.n_seeds):
        result = run_one_seed(
            seed=seed,
            proxy_timesteps=args.proxy_timesteps,
            n_trajectories=args.n_trajectories,
            n_pairs=args.n_pairs,
            rm_epochs=args.rm_epochs,
            rlhf_timesteps=args.rlhf_timesteps,
            eval_episodes=args.eval_episodes,
            device=device,
        )
        all_results.append(result)

    out_path = os.path.join(args.out_dir, "parametric_comparison.png")
    make_figure(all_results, out_path)

    # Summary
    print(f"\nSummary across {args.n_seeds} seeds:")
    for key, label in [
        ("proxy_gt",   "Proxy GT return  "),
        ("round1_gt",  "Round-1 GT return"),
        ("round2_gt",  "Round-2 GT return"),
        ("round1_r1t", "Round-1 r_1_term "),
        ("round2_r1t", "Round-2 r_1_term "),
    ]:
        vals = np.array([r[key] for r in all_results])
        print(f"  {label}: {vals.mean():.3f} ± {vals.std():.3f}")
    print(f"\nPlot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
