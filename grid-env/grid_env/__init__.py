from gym.envs.registration import register

register(
    id='grid-v0',
    entry_point='grid_env.envs:GridEnv',
)
