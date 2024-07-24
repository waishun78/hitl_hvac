from gymnasium.envs.registration import register
from gym_examples.gymnasium_aircon_env import AirconEnvironment
from gym_examples.hitl_env import HITLAirconEnvironment

register(
    id="AirconEnvironment-v0",
    entry_point="gym_examples:AirconEnvironment",
    max_episode_steps=300,
)

register(
    id="HITLAirconEnvironment-v0",
    entry_point="gym_examples:HITLAirconEnvironment",
    max_episode_steps=300,
)