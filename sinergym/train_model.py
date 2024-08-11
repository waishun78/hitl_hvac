import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import sinergym
from sinergym.utils.wrappers import NormalizeAction, NormalizeObservation

os.environ['EPLUS_PATH'] = '/usr/local/EnergyPlus-23-1-0'
pyenergyplus_path = '/usr/local/EnergyPlus-23-1-0/pyenergyplus'
os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':' + pyenergyplus_path

def make_env():
    env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
    env = NormalizeAction(env)
    env = NormalizeObservation(env)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)

# Training for a single episode
obs = env.reset()
done = False
episode_reward = 0
timesteps = []
mean_rewards = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    model.learn(total_timesteps=1)
    timesteps.append(model.num_timesteps)
    mean_rewards.append(episode_reward)

plt.figure(figsize=(10, 6))
plt.plot(timesteps, mean_rewards)
plt.title("PPO Learning Curve for a Single Episode")
plt.xlabel("Timesteps")
plt.ylabel("Cumulative Reward")
plt.savefig("ppo_training.png")
plt.show()

model.save("ppo_model")
env.close()
