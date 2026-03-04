import argparse
import glob
import os
import re
import gymnasium
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

import grid_world_env
from grid_world_env.wrappers import RelativePosition


def get_models():
    models = sorted(glob.glob("models/*"))
    return [m for m in models if os.path.isfile(m)]


def select_model():
    """List models and prompt the user to pick one by number."""
    models = get_models()
    if not models:
        print("No saved models found in models/")
        return None
    print("Saved models:")
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
    """Extract reward config and mask setting from auto-generated model filename."""
    name = os.path.basename(model_path)
    config = {}
    match = re.search(r"r0s([-\d.]+)_r0t([-\d.]+)_r1s([-\d.]+)_r1t([-\d.]+)", name)
    if match:
        config["reward_0_step"] = float(match.group(1))
        config["reward_0_terminal"] = float(match.group(2))
        config["reward_1_step"] = float(match.group(3))
        config["reward_1_terminal"] = float(match.group(4))
    config["mask_actions"] = "_masked" in name
    return config


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO on GridWorld")
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved model (without .zip)")
    parser.add_argument("--list", action="store_true", help="List all saved models")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--record", action="store_true", help="Record episodes as mp4 videos")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory to save recorded videos")
    parser.add_argument("--reward-0-step", type=float, default=None, help="Per-step reward for func_0")
    parser.add_argument("--reward-0-terminal", type=float, default=None, help="Terminal reward for func_0")
    parser.add_argument("--reward-1-step", type=float, default=None, help="Per-step reward for func_1")
    parser.add_argument("--reward-1-terminal", type=float, default=None, help="Terminal reward for func_1")
    parser.add_argument("--mask-actions", action="store_true", default=None, help="Enable action masking (auto-detected from model name)")
    parser.add_argument("--relative-reward", action="store_true", help="Use relative position as reward")
    parser.add_argument("--loop-detection", action="store_true", help="Switch reward function when looping behavior is detected")
    parser.add_argument("--loop-window", type=int, default=10, help="Number of recent steps to check for looping (default: 10)")
    parser.add_argument("--loop-grace-period", type=int, default=5, help="Number of loop detections to tolerate before switching reward function (default: 5)")

    args = parser.parse_args()

    if args.list:
        models = get_models()
        if not models:
            print("No saved models found in models/")
        else:
            print("Saved models:")
            for i, path in enumerate(models, 1):
                name = os.path.basename(path).replace(".zip", "")
                print(f"  {i}. {name}")
        return

    if args.model_path is None:
        args.model_path = select_model()
        if args.model_path is None:
            return

    # Auto-detect config from model filename
    parsed = parse_config_from_path(args.model_path)
    if parsed:
        print(f"Auto-detected config: {parsed}")

    reward_0_step = args.reward_0_step if args.reward_0_step is not None else parsed.get("reward_0_step", 1)
    reward_0_terminal = args.reward_0_terminal if args.reward_0_terminal is not None else parsed.get("reward_0_terminal", 100)
    reward_1_step = args.reward_1_step if args.reward_1_step is not None else parsed.get("reward_1_step", -1)
    reward_1_terminal = args.reward_1_terminal if args.reward_1_terminal is not None else parsed.get("reward_1_terminal", 100)
    mask_actions = args.mask_actions if args.mask_actions is not None else parsed.get("mask_actions", False)

    render_mode = "rgb_array" if args.record else "human"
    reward_mode = "relative" if args.relative_reward else "default"
    env = gymnasium.make(
        "grid_world_env/GridWorld-v0",
        max_episode_steps=100,
        render_mode=render_mode,
        reward_0_step=reward_0_step,
        reward_0_terminal=reward_0_terminal,
        reward_1_step=reward_1_step,
        reward_1_terminal=reward_1_terminal,
        reward_mode=reward_mode,
        loop_detection=args.loop_detection,
        loop_window=args.loop_window,
        loop_grace_period=args.loop_grace_period,
    )
    print(f"Using reward mode: {reward_mode}, loop detection: {args.loop_detection} (window={args.loop_window}, grace={args.loop_grace_period})")
    if mask_actions:
        env = ActionMasker(env, lambda e: e.get_wrapper_attr("action_masks")())
    env = RelativePosition(env)

    if args.record:
        env = RecordVideo(env, video_folder=args.video_dir, episode_trigger=lambda _: True)
        print(f"Recording videos to {args.video_dir}/")

    if mask_actions:
        model = MaskablePPO.load(args.model_path)
    else:
        model = PPO.load(args.model_path)
    print(f"Loaded {'MaskablePPO' if mask_actions else 'PPO'} model from {args.model_path}")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward = 0
        done = False

        while not done:
            if mask_actions:
                action, _ = model.predict(obs, deterministic=True, action_masks=env.get_wrapper_attr("action_masks")())
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            print(total_reward)
            done = terminated or truncated

        print(f"Episode {ep + 1}: total reward = {total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()
