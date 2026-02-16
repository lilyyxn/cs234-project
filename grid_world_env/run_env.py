# ref: https://gymnasium.farama.org/introduction/basic_usage/
from grid_world_env.envs.grid_world import GridWorldEnv
import gymnasium
import grid_world_env

env = gymnasium.make('grid_world_env/GridWorld-v0', max_episode_steps=100, render_mode="human")

# initial state
obs, info = env.reset(seed=2)
env.render()

episode_over = False
total_reward = 0
i = 0
while not episode_over:
    i += 1
    print("Episode", i)
    action = env.action_space.sample()  # random action
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    total_reward += reward
    print("total rewards", total_reward)
    episode_over = terminated or truncated
print(f"Episode finished! Total reward: {total_reward}")
env.close()