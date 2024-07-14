from gymnasium.envs.registration import register
from gym_examples.gymnasium_aircon_env import AirconEnvironment

register(
    id="AirconEnvironment-v0",
    entry_point="gym_examples:AirconEnvironment",
    max_episode_steps=300,
)