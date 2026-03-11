"""
Run the RLHF pipeline across multiple seeds and aggregate results.
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run RLHF across multiple seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Seeds to run")
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--n-preferences", type=int, default=500)
    parser.add_argument("--n-trajectories", type=int, default=500)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--reward-model-epochs", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="models/rlhf_multiseed")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    all_results = {}

    for seed in args.seeds:
        seed_dir = os.path.join(args.save_dir, f"seed_{seed}")
        print(f"\n{'#' * 60}")
        print(f"# Running seed {seed}")
        print(f"{'#' * 60}\n")

        cmd = [
            sys.executable, "grid_world_env/train_rlhf.py",
            "--seed", str(seed),
            "--n-rounds", str(args.n_rounds),
            "--n-preferences", str(args.n_preferences),
            "--n-trajectories", str(args.n_trajectories),
            "--timesteps", str(args.timesteps),
            "--reward-model-epochs", str(args.reward_model_epochs),
            "--save-dir", seed_dir,
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: Seed {seed} failed with return code {result.returncode}")
            continue

        # Load results
        results_path = os.path.join(seed_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                all_results[seed] = json.load(f)

    # Save aggregated results
    agg_path = os.path.join(args.save_dir, "all_results.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 70}")
    print("Multi-seed Results Summary")
    print(f"{'=' * 70}")

    n_rounds = args.n_rounds
    header = f"{'Seed':>6}"
    for r in range(n_rounds + 1):
        header += f"  {'Round ' + str(r):>12}"
    print(header)
    print("-" * len(header))

    for seed in sorted(all_results.keys()):
        row = f"{seed:>6}"
        for r_data in all_results[seed]:
            row += f"  {r_data['gt_return_mean']:>12.1f}"
        print(row)

    print(f"\nAggregated results saved to {agg_path}")


if __name__ == "__main__":
    main()
