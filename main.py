import time


if __name__ == "__main__":
    # from gym_examples.gymnasium_aircon_env import AirconEnvironment
    # from stable_baselines3.common.env_checker import check_env

    # print("--------------Testing: GYMNASIUM AIRCON ENVIRONMENT--------------")
    # env = AirconEnvironment(True, False, 1, 1)
    # check_env(env)
    # for i in range(100):
    #     env.step(40)
    #     time.sleep(1)
    # from gym_examples.torchrl_aircon_env import AirconEnvironment
    # from torchrl.envs.utils import check_env_specs
    # import torch
    # from tensordict import TensorDict

    # print("--------------Testing: TORCHRL AIRCON ENVIRONMENT--------------")
    # env = AirconEnvironment(30, True, 1, 1) # create environment object
    # check_env_specs(env)

    # # Simulate applying a constant action 24
    # rollout = env.rollout(10, policy=lambda _: TensorDict({"action": 24.0}, batch_size=torch.Size()))
    # print("Observation: ", rollout["observation"])
    from gym_examples.utils.population import PopulationSimulation
    from gym_examples.hitl_mdp_env import HITLAirconEnvironment

    sim = PopulationSimulation(1, 0.2, 2, 0.3, 100,30)

    env = HITLAirconEnvironment(sim, False, False, 1)
    print(env.reset())
    print(env.step(27))
    print(env.step(26))
    # print(f'Step:{sim.step()}')
    # print(sim.get_comfort_profile_df())

    # from pythermalcomfort.models import pmv_ppd
    # print(pmv_ppd(27,27,0.1,50,0.8,2,5, standard="ASHRAE"))
