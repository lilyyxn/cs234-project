import argparse
import glob
import os
import re
import gymnasium
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

import grid_world_env
from grid_world_env.wrappers import RelativePosition


def get_models():
    models = sorted(glob.glob("models/dqn_*"))
    return [m for m in models if os.path.isfile(m)]


def select_model():
    """List DQN models and prompt the user to pick one by number."""
    models = get_models()
    if not models:
        print("No saved DQN models found in models/")
        return None
    print("Saved DQN models:")
    for i, path in enumerate(models, 1):
        name = os.path.basename(path).replace(".zip", "")
        print(f"  {i}. {name}")
    print()
    while True:
        choice = input("Select a model (number): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print(f"Invalid choice. Enter a number between 1 and {len(models)}.")


def parse_config_from_path(model_path):
    """Extract reward config from auto-generated DQN model filename."""
    name = os.path.basename(model_path)
    config = {}
    match = re.search(r"r0s([-\d.]+)_r0t([-\d.]+)_r1s([-\d.]+)_r1t([-\d.]+)", name)
    if match:
        config["reward_0_step"] = float(match.group(1))
        config["reward_0_terminal"] = float(match.group(2))
        config["reward_1_step"] = float(match.group(3))
        config["reward_1_terminal"] = float(match.group(4))
    return config


def run_episodes(model, env, n_episodes, seed=0):
    """Run n_episodes and return a list of total rewards."""
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN on GridWorld")
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved model (without .zip)")
    parser.add_argument("--list", action="store_true", help="List all saved DQN models")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--record", action="store_true", help="Record episodes as mp4 videos")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory to save recorded videos")
    parser.add_argument("--reward-0-step", type=float, default=None, help="Per-step reward for func_0")
    parser.add_argument("--reward-0-terminal", type=float, default=None, help="Terminal reward for func_0")
    parser.add_argument("--reward-1-step", type=float, default=None, help="Per-step reward for func_1")
    parser.add_argument("--reward-1-terminal", type=float, default=None, help="Terminal reward for func_1")
    args = parser.parse_args()

    if args.list:
        models = get_models()
        if not models:
            print("No saved DQN models found in models/")
        else:
            print("Saved DQN models:")
            for i, path in enumerate(models, 1):
                name = os.path.basename(path).replace(".zip", "")
                print(f"  {i}. {name}")
        return

    if args.model_path is None:
        args.model_path = select_model()
        if args.model_path is None:
            return

    parsed = parse_config_from_path(args.model_path)
    if parsed:
        print(f"Auto-detected config: {parsed}")

    reward_0_step = args.reward_0_step if args.reward_0_step is not None else parsed.get("reward_0_step", 1)
    reward_0_terminal = args.reward_0_terminal if args.reward_0_terminal is not None else parsed.get("reward_0_terminal", 100)
    reward_1_step = args.reward_1_step if args.reward_1_step is not None else parsed.get("reward_1_step", -1)
    reward_1_terminal = args.reward_1_terminal if args.reward_1_terminal is not None else parsed.get("reward_1_terminal", 100)

    render_mode = "rgb_array" if args.record else "human"
    env = gymnasium.make(
        "grid_world_env/GridWorld-v0",
        max_episode_steps=100,
        render_mode=render_mode,
        reward_0_step=reward_0_step,
        reward_0_terminal=reward_0_terminal,
        reward_1_step=reward_1_step,
        reward_1_terminal=reward_1_terminal,
    )
    env = RelativePosition(env)

    if args.record:
        env = RecordVideo(env, video_folder=args.video_dir, episode_trigger=lambda _: True)
        print(f"Recording videos to {args.video_dir}/")

    model = DQN.load(args.model_path)
    print(f"Loaded DQN model from {args.model_path}")

    rewards = run_episodes(model, env, n_episodes=args.episodes, seed=args.seed)
    for ep, total_reward in enumerate(rewards, 1):
        print(f"Episode {ep}: total reward = {total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()
