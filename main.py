if __name__ == "__main__":
    from gym_examples.gymnasium_aircon_env import AirconEnvironment
    from stable_baselines3.common.env_checker import check_env

    print("--------------Testing: GYMNASIUM AIRCON ENVIRONMENT--------------")
    env = AirconEnvironment(False, 30, 1, 1)
    check_env(env)

    from gym_examples.torchrl_aircon_env import AirconEnvironment
    from torchrl.envs.utils import check_env_specs
    import torch
    from tensordict import TensorDict

    print("--------------Testing: TORCHRL AIRCON ENVIRONMENT--------------")
    env = AirconEnvironment(30, True, 1, 1) # create environment object
    check_env_specs(env)

    # Simulate applying a constant action 24
    rollout = env.rollout(10, policy=lambda _: TensorDict({"action": 24.0}, batch_size=torch.Size()))
    print("Observation: ", rollout["observation"])