import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
from gym_examples.utils.population import PopulationSimulation
from models.HITLDQNAgent import HITLDQNAgent

# Needed:Import gym environment (assumed to be already registered)
import gym_examples

LOAD_FILEPATH = "runs/saved_models/hitl_dqnv0"

def plot_durations(rewards_ls, show_result=False):
    plt.figure(1)
    
    if not show_result:
        plt.clf()
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Daily Cummulative Reward', fontsize=14)
    plt.plot(rewards_ls)
    # plt.plot(fixed_rsts)
    plt.legend(["RL policy", "Set point control"])
    
    plt.pause(0.001)
    if not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)
    else:
        display.display(plt.gcf())

import time
from models.BaseAgent import BaseAgent
import gymnasium as gym
import torch
import numpy as np
from gym_examples.utils.population import PopulationSimulation
from models.HITLDQNAgent import HITLDQNAgent
from models.HITLDRQNAgent import HITLDRQNAgent
# Initialise training environment

def train_w_reset_h(agent:HITLDRQNAgent, env:gym.Env, episodes: int, device, is_exploring:bool):
    """Train agent using gym environment while using replay memory for Recurrent Neural Network"""

    rewards = []

    agent.reset_replay_memory() # Same train function has the same episodic memory
    agent.set_exploring(True)

    for i_episode in range(episodes):
        print(f'######------------------------------------EPISODE {i_episode}------------------------------------######')
        observation, info = env.reset()
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        ep_rewards = 0
        is_done = False

        while not is_done:
            action = agent.select_action(state)
            observation, reward, terminated, truncated, info = env.step(action.item())
            reward = torch.tensor([np.float32(reward)], device=device)
            ep_rewards += reward

            is_done = terminated or truncated
            next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            agent.memorize(state, action, next_state, reward)
            state = next_state
            agent.optimize_model()


        rewards.append(ep_rewards.item())

        agent.reset_hidden_state() # Reset the hidden state of the agent

    return agent, rewards

def train(agent:BaseAgent, env:gym.Env, episodes: int, device, is_exploring:bool):
    """Train agent using gym environment using replay memory"""

    rewards = []

    agent.reset_replay_memory() # Same train function has the same episodic memory
    agent.set_exploring(True)

    for i_episode in range(episodes):
        print(f'######------------------------------------EPISODE {i_episode}------------------------------------######')
        observation, info = env.reset()
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        ep_rewards = 0
        is_done = False

        i = 0
        while not is_done:
            action = agent.select_action(state)
            observation, reward, terminated, truncated, info = env.step(action.item())
            reward = torch.tensor([np.float32(reward)], device=device)
            ep_rewards += reward
            
            is_done = terminated or truncated
            next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            agent.memorize(state, action, next_state, reward)
            state = next_state
            agent.optimize_model()

        rewards.append(ep_rewards.item())
            
    return agent, rewards

if __name__ == "__main__":
    print("######------------------------------------Starting training...------------------------------------######")
    # set up matplotlib
    from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    population = PopulationSimulation(1, 0.3, 0.5, 2, 100, 50)
    # w_energy, w_usercomfort is from generated from a comfort score of range (-40 - 0) and power score (-1000 - -300)
    env = gym.make("AirconEnvironment-v0", population_simulation=population, is_render=False, check_optimal=False ,w_usercomfort=20, w_energy=1)
    print("######------------------------------------Resetting environment...------------------------------------######")
    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    agent = HITLDQNAgent(n_observations, n_actions, device)

    episode_durations = []

    NUM_EPISODES = 100 if torch.cuda.is_available() or torch.backends.mps.is_available() else 50 # Number of days (each episode is a day)

    rewards = train(agent, env, NUM_EPISODES, device, True)

    agent.save_model(LOAD_FILEPATH)

    plot_durations(rewards, show_result=True)
    plt.ioff()
    plt.show()

    print('Complete')
