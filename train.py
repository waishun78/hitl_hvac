import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
from models.HITLDQNAgent import HITLDQNAgent

# Needed:Import gym environment (assumed to be already registered)
import gym_examples

LOAD_FILEPATH = "runs/saved_models/hitl_dqnv0"

# set up matplotlib
from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

env = gym.make("AirconEnvironment-v0", is_render=False, alpha=1, beta=1)
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

agent = HITLDQNAgent(n_observations, n_actions, device)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    
    if not show_result:
        plt.clf()
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Daily Cummulative Reward', fontsize=14)
    plt.plot(rewards_ls)
    # plt.plot(fixed_rsts)
    plt.legend(["RL policy", "Set point control"])
    
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

NUM_EPISODES = 200 if torch.cuda.is_available() or torch.backends.mps.is_available() else 50 # Number of days (each episode is a day)
rewards_ls = []

for i_episode in range(NUM_EPISODES):
    observation, info = env.reset()
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    accum_rewards = 0
    done = False

    i = 0
    while not done:
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([np.float32(reward)], device=device)
        accum_rewards += reward
        done = terminated or truncated
        next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        agent.memorize(state, action, next_state, reward)
        state = next_state
        agent.optimize_model()

    rewards_ls.append(accum_rewards.item())

agent.save_model(LOAD_FILEPATH)

plot_durations(show_result=True)
plt.ioff()
plt.show()

print('Complete')
