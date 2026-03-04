# ref: https://gymnasium.farama.org/introduction/basic_usage/
import argparse
from grid_world_env.envs.grid_world import GridWorldEnv
import gymnasium
import grid_world_env

parser = argparse.ArgumentParser(description="Run GridWorld environment")
parser.add_argument("--relative-reward", action="store_true", help="Use relative position (negative Manhattan distance) as reward instead of +1/-1")
parser.add_argument("--loop-detection", action="store_true", help="Switch reward function when looping behavior is detected")
parser.add_argument("--loop-window", type=int, default=10, help="Number of recent steps to check for looping (default: 10)")
parser.add_argument("--loop-grace-period", type=int, default=5, help="Number of loop detections to tolerate before switching reward function (default: 5)")
args = parser.parse_args()

reward_mode = "relative" if args.relative_reward else "default"
print(f"Using reward mode: {reward_mode}, loop detection: {args.loop_detection} (window={args.loop_window}, grace={args.loop_grace_period})")

env = gymnasium.make('grid_world_env/GridWorld-v0', max_episode_steps=100, render_mode="human",
                     reward_mode=reward_mode, loop_detection=args.loop_detection,
                     loop_window=args.loop_window, loop_grace_period=args.loop_grace_period)

# initial state
obs, info = env.reset(seed=2)
env.render()

episode_over = False
total_reward = 0
i = 0
while not episode_over:
    i += 1
    print("Step", i)
    action = env.action_space.sample()  # random action
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    total_reward += reward
    print("total rewards", total_reward)
    episode_over = terminated or truncated
print(f"Episode finished! Total reward: {total_reward}")
env.close()