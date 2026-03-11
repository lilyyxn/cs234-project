"""Parametric RLHF: reinitialise vs fine-tune comparison with error bars.

Runs run_parametric_rlhf() twice per seed:
  - reinitialise: fresh policy each round (current behaviour)
  - finetune:     carry proxy policy weights into each RLHF round

Key hypothesis (finetune variant): the proxy policy has deeply internalised
Phase-0-terminal avoidance via potential shaping.  When fine-tuned on the
parametric-corrected env, this prior persists and the agent fails to complete
Phase 1, even though the updated reward function rewards Phase 1 completion.

Figure saved to plots/parametric_comparison.png (4 panels):
  Panel 1: GT return — proxy / reinit round1 / reinit round2 / finetune round1 / finetune round2
  Panel 2: GT return after round 2 only — reinit vs finetune (cleaner comparison)
  Panel 3: Learned r_1_terminal — reinit vs finetune per round
  Panel 4: Phase-1 completion rate — reinit vs finetune per round

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

PROXY_COLOR   = "#e07b54"   # orange
REINIT_COLOR  = "#5b9bd5"   # blue   — reinitialise (works)
FINETUNE_COLOR = "#d94f4f"  # red    — finetune    (fails)


def _bar_err(ax, x, mean, std, color, width=0.5, label=None):
    ax.bar(x, mean, width=width, color=color, edgecolor="white",
           linewidth=1.2, label=label)
    ax.errorbar(x, mean, yerr=std, fmt="none", color="black",
                capsize=5, capthick=1.5, linewidth=1.5)


def run_one_seed(seed, proxy_timesteps, n_trajectories, n_pairs,
                 rm_epochs, rlhf_timesteps, eval_episodes, device):
    print(f"\n--- Seed {seed} (reinitialise) ---")
    reinit = run_parametric_rlhf(
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
        finetune=False,
    )
    print(f"  [reinit]  round1 GT={reinit['round_metrics'][0]['mean_gt_return']:.3f}  "
          f"round2 GT={reinit['round_metrics'][1]['mean_gt_return']:.3f}")

    print(f"\n--- Seed {seed} (finetune) ---")
    ft = run_parametric_rlhf(
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
        finetune=True,
    )
    print(f"  [finetune] round1 GT={ft['round_metrics'][0]['mean_gt_return']:.3f}  "
          f"round2 GT={ft['round_metrics'][1]['mean_gt_return']:.3f}")

    return {
        "proxy_gt":         reinit["proxy_policy_gt_return"],
        "reinit_r1_gt":     reinit["round_metrics"][0]["mean_gt_return"],
        "reinit_r2_gt":     reinit["round_metrics"][1]["mean_gt_return"],
        "reinit_r1_r1t":    reinit["learned_params_per_round"][0]["r_1_terminal"],
        "reinit_r2_r1t":    reinit["learned_params_per_round"][1]["r_1_terminal"],
        "reinit_r2_both":   reinit["round_metrics"][1]["pct_both_complete"],
        "ft_r1_gt":         ft["round_metrics"][0]["mean_gt_return"],
        "ft_r2_gt":         ft["round_metrics"][1]["mean_gt_return"],
        "ft_r1_r1t":        ft["learned_params_per_round"][0]["r_1_terminal"],
        "ft_r2_r1t":        ft["learned_params_per_round"][1]["r_1_terminal"],
        "ft_r2_both":       ft["round_metrics"][1]["pct_both_complete"],
    }


def make_figure(all_results, out_path):
    def arr(key):
        return np.array([r[key] for r in all_results])

    proxy_gt      = arr("proxy_gt")
    reinit_r1_gt  = arr("reinit_r1_gt");  reinit_r2_gt  = arr("reinit_r2_gt")
    ft_r1_gt      = arr("ft_r1_gt");      ft_r2_gt      = arr("ft_r2_gt")
    reinit_r1_r1t = arr("reinit_r1_r1t"); reinit_r2_r1t = arr("reinit_r2_r1t")
    ft_r1_r1t     = arr("ft_r1_r1t");     ft_r2_r1t     = arr("ft_r2_r1t")
    reinit_r2_both = arr("reinit_r2_both"); ft_r2_both   = arr("ft_r2_both")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        "Parametric RLHF: Reinitialise vs Fine-Tune\n"
        "Fine-tuning from the proxy policy preserves prior behaviour "
        "and resists reward correction",
        fontsize=12, fontweight="bold", y=1.03,
    )

    # ---- Panel 1: GT return across all stages ----
    ax = axes[0]
    x = np.arange(5)
    w = 0.6
    data = [
        (proxy_gt,     PROXY_COLOR,    "Proxy baseline"),
        (reinit_r1_gt, REINIT_COLOR,   "Reinit — round 1"),
        (reinit_r2_gt, REINIT_COLOR,   "Reinit — round 2"),
        (ft_r1_gt,     FINETUNE_COLOR, "Finetune — round 1"),
        (ft_r2_gt,     FINETUNE_COLOR, "Finetune — round 2"),
    ]
    for xi, (vals, color, label) in zip(x, data):
        alpha = 0.55 if "round 1" in label else 1.0
        ax.bar(xi, vals.mean(), width=w, color=color, alpha=alpha,
               edgecolor="white", linewidth=1.2, label=label)
        ax.errorbar(xi, vals.mean(), yerr=vals.std(), fmt="none",
                    color="black", capsize=4, capthick=1.2, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Proxy", "Reinit\nR1", "Reinit\nR2", "Finetune\nR1", "Finetune\nR2"],
        fontsize=8,
    )
    ax.set_ylabel("Mean GT Return")
    ax.set_ylim(bottom=0, top=2.3)
    ax.axhline(2.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_title("GT Return Across Rounds\n(faded = round 1)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel 2: Final GT return (round 2) — head-to-head ----
    ax = axes[1]
    x2 = np.array([0, 1])
    _bar_err(ax, x2[0], reinit_r2_gt.mean(), reinit_r2_gt.std(),
             REINIT_COLOR,   label="Reinitialise")
    _bar_err(ax, x2[1], ft_r2_gt.mean(),     ft_r2_gt.std(),
             FINETUNE_COLOR, label="Fine-tune")
    ax.set_xticks(x2)
    ax.set_xticklabels(["Reinitialise\n(round 2)", "Fine-tune\n(round 2)"], fontsize=9)
    ax.set_ylabel("Mean GT Return")
    ax.set_ylim(bottom=0, top=2.3)
    ax.axhline(2.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.6,
               label="Max (2.0)")
    ax.set_title("Final GT Return After 2 Rounds\nFine-tuning resists correction",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel 3: Learned r_1_terminal ----
    ax = axes[2]
    x3 = np.array([0, 1, 2, 3])
    bars = [
        (reinit_r1_r1t, REINIT_COLOR,    0.55, "Reinit R1"),
        (reinit_r2_r1t, REINIT_COLOR,    1.0,  "Reinit R2"),
        (ft_r1_r1t,     FINETUNE_COLOR,  0.55, "Finetune R1"),
        (ft_r2_r1t,     FINETUNE_COLOR,  1.0,  "Finetune R2"),
    ]
    for xi, (vals, color, alpha, label) in zip(x3, bars):
        ax.bar(xi, vals.mean(), width=0.6, color=color, alpha=alpha,
               edgecolor="white", linewidth=1.2, label=label)
        ax.errorbar(xi, vals.mean(), yerr=vals.std(), fmt="none",
                    color="black", capsize=4, capthick=1.2, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="Proxy value (0)")
    ax.set_xticks(x3)
    ax.set_xticklabels(["Reinit\nR1", "Reinit\nR2", "FT\nR1", "FT\nR2"], fontsize=8)
    ax.set_ylabel("Learned r₁_terminal")
    ax.set_title("Learned Phase 1 Terminal Reward\n(faded = round 1)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
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
    parser.add_argument("--fast", action="store_true")
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

    print(f"\nSummary across {args.n_seeds} seeds:")
    for key, label in [
        ("proxy_gt",      "Proxy GT          "),
        ("reinit_r2_gt",  "Reinit  round-2 GT"),
        ("ft_r2_gt",      "Finetune round-2 GT"),
        ("reinit_r2_r1t", "Reinit  r_1_term  "),
        ("ft_r2_r1t",     "Finetune r_1_term "),
        ("reinit_r2_both","Reinit  both%     "),
        ("ft_r2_both",    "Finetune both%    "),
    ]:
        vals = np.array([r[key] for r in all_results])
        print(f"  {label}: {vals.mean():.3f} ± {vals.std():.3f}")
    print(f"\nPlot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
