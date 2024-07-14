import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
from gym_examples.utils.population import PopulationSimulation
from models.HITLDQNAgent import HITLDQNAgent

# Needed:Import gym environment (assumed to be already registered)
import gym_examples

# set up matplotlib
from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

ENV = "AirconEnvironment-v0"
LOAD_FILEPATH = "runs/saved_models/hitl_dqnv0"
TEST_EPISODES = 20

population = PopulationSimulation(2, 0.3, 1, 2, 100, 50)
env = gym.make("AirconEnvironment-v0", population_simulation=population, is_render=False, check_optimal=False ,w_usercomfort=20, w_energy=1)
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

agent = HITLDQNAgent(n_observations, n_actions, device)
agent.set_learning(False) # No more random policy exploration

agent.load_model(LOAD_FILEPATH)


def evaluate_model(model, num_episodes=100):
    total_rewards = []
    for i_episode in range(TEST_EPISODES):
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

            state = next_state

        total_rewards.append(accum_rewards.item())
    return total_rewards

num_test_episodes = 100

#TODO: Update this for all models evaluated
hitl_energy_rewards = evaluate_model(agent, num_episodes=num_test_episodes)

plt.figure(figsize=(10, 5))
plt.plot(hitl_energy_rewards, label='hitl+energy')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Comparison (Using Average Reward)')
plt.legend()

display.display(plt.gcf())

plt.ioff()
plt.show()