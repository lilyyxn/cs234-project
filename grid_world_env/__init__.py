from gymnasium.envs.registration import register

register(
    id="grid_world_env/GridWorld-v0",
    entry_point="grid_world_env.envs:GridWorldEnv",
)
