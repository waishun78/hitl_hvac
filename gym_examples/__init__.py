from gymnasium.envs.registration import register
from gym_examples.hitl_pomdp_energy_env import AirconEnvironment
from gym_examples.hitl_mdp_env import HITLAirconEnvironment
from gym_examples.hitl_pomdp_env import HITLPOMDPAirconEnvironment
from gym_examples.energy_only_env import EnergyOnlyAirconEnvironment

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

register(
    id="HITLAPOMDPirconEnvironment-v0",
    entry_point="gym_examples:HITLPOMDPAirconEnvironment",
    max_episode_steps=300,
)

register(
    id="EnergyOnlyAirconEnvironment-v0",
    entry_point="gym_examples:EnergyOnlyAirconEnvironment",
    max_episode_steps=300,
)