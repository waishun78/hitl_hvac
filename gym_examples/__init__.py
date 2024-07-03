from gymnasium.envs.registration import register

register(
    id="AirconEnvironment-v0",
    entry_point="AirconEnvironment",
     max_episode_steps=300,
)