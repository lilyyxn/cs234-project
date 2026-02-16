# ref: https://gymnasium.farama.org/introduction/basic_usage/
from grid_world_env.envs.grid_world import GridWorldEnv

env = GridWorldEnv(render_mode="human")

# initial state
obs, info = env.reset()
env.render()

episode_over = False
total_reward = 0
while not episode_over:
    action = env.action_space.sample()  # random action
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    total_reward += reward
    episode_over = terminated or truncated
print(f"Episode finished! Total reward: {total_reward}")
env.close()