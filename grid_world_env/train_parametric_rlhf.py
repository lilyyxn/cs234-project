"""Parametric RLHF pipeline for the reward hacking GridWorld.

Unlike the neural RLHF approach (train_rlhf.py), this learns 4 interpretable
reward parameters from preferences and plugs them directly back into make_env().
The corrected reward function therefore lives INSIDE the environment, exactly
mirroring how the original manual reward switching works.

Key claim (from mentor feedback): even with parametric preference correction,
the maximum achievable return under the learned reward may still make Phase 1
avoidance rational, revealing a fundamental limitation of RLHF correction when
the preference data is dominated by Phase-0-only trajectories.

Pipeline:
  1. Train proxy policy on the original misspecified reward.
  2. For each RLHF round:
       a. Collect trajectories from current policy.
       b. Label preference pairs via Boltzmann GT labeler.
       c. Train ParametricRewardModel (4 scalars) via Bradley-Terry BCE.
       d. Create a new env with the learned parameters (no wrapper).
       e. Re-train policy on the updated env.
  3. Evaluate and report learned parameters + GT return progression.

Usage:
  python grid_world_env/train_parametric_rlhf.py
  python grid_world_env/train_parametric_rlhf.py --proxy-timesteps 300000
"""

import argparse

import numpy as np
import torch

from grid_world_env.train_ppo_scratch import ActorCritic, make_env, train
from grid_world_env.rlhf.parametric_reward_model import (
    ParametricRewardModel,
    train_parametric_reward_model,
)
from grid_world_env.rlhf.preference_data import collect_trajectories, generate_preference_pairs
from grid_world_env.train_rlhf import evaluate_policy


def run_parametric_rlhf(
    proxy_timesteps: int = 300_000,
    n_rlhf_rounds: int = 2,
    n_trajectories_per_round: int = 150,
    n_pairs_per_round: int = 300,
    reward_model_epochs: int = 100,
    rlhf_timesteps_per_round: int = 150_000,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    lr: float = 3e-4,
    eval_episodes: int = 100,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
    finetune: bool = False,
) -> dict:
    """Run parametric RLHF: learn reward params from preferences, retrain on updated env.

    Args:
        finetune: if True, each RLHF round fine-tunes from the previous round's
                  policy (init_policy=prev_policy) rather than reinitialising
                  from scratch.  This tests whether a policy that has already
                  internalised the proxy exploit can be corrected by the updated
                  reward function, or whether its prior behaviour persists.

    Returns dict with:
        proxy_policy_gt_return, rlhf_policy_gt_return, hiding_incentive,
        learned_params_per_round, final_learned_params, round_metrics,
        proxy_policy, rlhf_policy
    """
    def _log(msg):
        if verbose:
            print(msg)

    proxy_env = make_env(
        use_potential_shaping=True,
        reward_0_step=0, reward_0_terminal=1,
        reward_1_step=-2, reward_1_terminal=0,
    )

    # Phase 1: train proxy policy (identical to train_rlhf.py)
    _log("\n=== Phase 1: Training on proxy reward ===")
    policy = train(
        proxy_env, total_timesteps=proxy_timesteps,
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
        gamma=gamma, lr=lr, seed=seed, device=device,
    )

    proxy_metrics = evaluate_policy(policy, proxy_env,
                                    n_episodes=eval_episodes, device=device)
    _log(f"  GT return:    {proxy_metrics['mean_gt_return']:.3f}")
    _log(f"  Proxy return: {proxy_metrics['mean_proxy_return']:.3f}")
    _log(f"  Phase-0 completion: {proxy_metrics['pct_phase0_complete']:.1%}")
    _log(f"  Both phases:        {proxy_metrics['pct_both_complete']:.1%}")

    # Phase 2: parametric RLHF rounds
    param_model = ParametricRewardModel()
    rlhf_policy = policy
    round_metrics = []
    learned_params_per_round = []

    for round_idx in range(n_rlhf_rounds):
        _log(f"\n=== Parametric RLHF Round {round_idx + 1}/{n_rlhf_rounds} ===")

        trajs = collect_trajectories(rlhf_policy, proxy_env,
                                     n=n_trajectories_per_round, device=device)
        gt_returns = [t.gt_return for t in trajs]
        _log(f"  Trajectories: {len(trajs)}, mean GT={np.mean(gt_returns):.2f}")

        # Log phase coverage — key diagnostic for bootstrapping problem
        n_phase1 = sum(1 for t in trajs if any(t.observations[:, 2] == 1.0))
        n_both   = sum(1 for t in trajs if t.gt_return >= 2.0)
        _log(f"  Trajectories entering Phase 1: {n_phase1}/{len(trajs)} "
             f"({n_phase1/len(trajs):.1%})")
        _log(f"  Trajectories completing both:  {n_both}/{len(trajs)} "
             f"({n_both/len(trajs):.1%})")

        pairs = generate_preference_pairs(trajs, n_pairs=n_pairs_per_round,
                                          seed=seed + round_idx)

        _log(f"  Training parametric reward model ({reward_model_epochs} epochs)...")
        train_parametric_reward_model(param_model, pairs,
                                      n_epochs=reward_model_epochs)

        learned = param_model.as_dict()
        learned_params_per_round.append(learned)
        _log(f"  Learned params:")
        _log(f"    r_0_step     = {learned['r_0_step']:.4f}  "
             f"(original: 0)")
        _log(f"    r_0_terminal = {learned['r_0_terminal']:.4f}  "
             f"(original: 1)")
        _log(f"    r_1_step     = {learned['r_1_step']:.4f}  "
             f"(original: -2)")
        _log(f"    r_1_terminal = {learned['r_1_terminal']:.4f}  "
             f"(original: 0)")

        # Compute break-even: at what r_1_terminal does completing Phase 1 beat staying?
        # Agent needs r_1_terminal + r_1_step * d > 0, where d = steps to reach target
        # For a random target on a 5x5 grid, expected Manhattan distance ≈ 4-5 steps
        # Break-even: r_1_terminal > -r_1_step * d_expected
        d_expected = 4.0
        break_even = -learned['r_1_step'] * d_expected
        completing_optimal = learned['r_1_terminal'] > break_even
        _log(f"  Phase 1 completion optimal? {completing_optimal} "
             f"(r_1_terminal={learned['r_1_terminal']:.3f} vs "
             f"break-even≈{break_even:.3f})")

        # Create updated env with learned parameters — no wrapper, params IN the env
        updated_env = make_env(
            use_potential_shaping=False,
            reward_0_step=learned['r_0_step'],
            reward_0_terminal=learned['r_0_terminal'],
            reward_1_step=learned['r_1_step'],
            reward_1_terminal=learned['r_1_terminal'],
        )

        init = rlhf_policy if finetune else None
        mode = "Fine-tuning" if finetune else "Re-training"
        _log(f"  {mode} policy on env with learned parameters...")
        rlhf_policy = train(
            updated_env, total_timesteps=rlhf_timesteps_per_round,
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
            gamma=gamma, lr=lr, seed=seed + round_idx + 1, device=device,
            init_policy=init,
        )

        round_m = evaluate_policy(rlhf_policy, proxy_env,
                                  n_episodes=eval_episodes, device=device)
        round_metrics.append(round_m)
        _log(f"  GT return={round_m['mean_gt_return']:.3f}, "
             f"phase-0={round_m['pct_phase0_complete']:.1%}, "
             f"both={round_m['pct_both_complete']:.1%}")

    rlhf_metrics = evaluate_policy(rlhf_policy, proxy_env,
                                   n_episodes=eval_episodes, device=device)

    hiding_incentive = (proxy_metrics["mean_proxy_return"]
                        - rlhf_metrics["mean_proxy_return"])

    _log(f"\n=== Summary ===")
    _log(f"  Proxy policy GT return: {proxy_metrics['mean_gt_return']:.3f}")
    _log(f"  RLHF  policy GT return: {rlhf_metrics['mean_gt_return']:.3f}")
    _log(f"  Hiding incentive:       {hiding_incentive:.3f}")
    final = param_model.as_dict()
    _log(f"  Final learned params:   r_0_step={final['r_0_step']:.3f}, "
         f"r_0_terminal={final['r_0_terminal']:.3f}, "
         f"r_1_step={final['r_1_step']:.3f}, "
         f"r_1_terminal={final['r_1_terminal']:.3f}")

    return {
        "proxy_policy_gt_return":    proxy_metrics["mean_gt_return"],
        "proxy_policy_proxy_return": proxy_metrics["mean_proxy_return"],
        "proxy_pct_phase0":          proxy_metrics["pct_phase0_complete"],
        "proxy_pct_both":            proxy_metrics["pct_both_complete"],
        "rlhf_policy_gt_return":     rlhf_metrics["mean_gt_return"],
        "rlhf_policy_proxy_return":  rlhf_metrics["mean_proxy_return"],
        "rlhf_pct_phase0":           rlhf_metrics["pct_phase0_complete"],
        "rlhf_pct_both":             rlhf_metrics["pct_both_complete"],
        "hiding_incentive":          hiding_incentive,
        "round_metrics":             round_metrics,
        "learned_params_per_round":  learned_params_per_round,
        "final_learned_params":      final,
        "proxy_policy":              policy,
        "rlhf_policy":               rlhf_policy,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parametric RLHF: learn reward params from preferences"
    )
    parser.add_argument("--proxy-timesteps",  type=int, default=300_000)
    parser.add_argument("--n-rlhf-rounds",    type=int, default=2)
    parser.add_argument("--n-trajectories",   type=int, default=150)
    parser.add_argument("--n-pairs",          type=int, default=300)
    parser.add_argument("--rm-epochs",        type=int, default=100)
    parser.add_argument("--rlhf-timesteps",   type=int, default=150_000)
    parser.add_argument("--eval-episodes",    type=int, default=100)
    parser.add_argument("--seed",             type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running parametric RLHF | device={device}")

    run_parametric_rlhf(
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


if __name__ == "__main__":
    main()
