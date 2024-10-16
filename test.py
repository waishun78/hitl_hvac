import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
from gym_examples.utils.population import PopulationSimulation
from models.HITLDQNAgent import HITLDQNAgent

# Needed:Import gym environment (assumed to be already registered)
import gym_examples


ENV = "AirconEnvironment-v0"
LOAD_FILEPATH = "runs/saved_models/hitl_dqnv0"
TEST_EPISODES = 20


# (Comment from Jetwei) Added this function, which plots a graph of actions taken against steps in each Episode, but question is how do evaluate that this is the optimal action for each time?
def plot_graph_of_actions_taken_against_steps(actions_taken_list, episode_number):
    x = np.arange(len(actions_taken_list))
    
    plt.clf()  # Clear the current figure
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Action taken (aka Temperature set)', fontsize=14)
    plt.plot(x, actions_taken_list)

    # Title for each graph indicating the episode number
    plt.title(f'Actions taken in Episode {episode_number}')

    plt.pause(0.001)  # Pause to allow the plot to update
    plt.show()


# (Comment from Jetwei) Added some corresponding code here in order to use the 'plot_graph_of_actions_taken_against_steps()'
# function I created above
def evaluate_model(agent, num_episodes=100):
    total_rewards = []

    for i_episode in range(TEST_EPISODES):
        actions_taken_throughout_the_episode = []

        print(f'######------------------------------------EPISODE {i_episode}------------------------------------######')
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

            actions_taken_throughout_the_episode.append(action)


        # Plotting graph of actions taken against steps throughout each Episode
        plot_graph_of_actions_taken_against_steps(actions_taken_throughout_the_episode, i_episode)


        print(f"Cumulative Reward for Episode {i_episode}: {accum_rewards}")
        print("\n")
        total_rewards.append(accum_rewards.item())

    return total_rewards


# ////////////////////////////////////////////////////////////////////////////////////////////


# ########################### #
#                             #
# Running the testing process #
#                             #
# ########################### #


#################
# Miscellaneous #
#################
print("######------------------------------------Starting testing...------------------------------------######")

# set up matplotlib
from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


#############################################################################
# Initiating the Air Con Gymnasium RL Environment and Population Simulation #
#############################################################################c:\Users\Jet Wei\Pictures\Screenshots\Screenshot 2024-10-02 012721.png
population = PopulationSimulation(2, 0.3, 1, 2, 100, 50)
env = gym.make(ENV, population_simulation=population, is_render=False, check_optimal=False ,w_usercomfort=20, w_energy=1)
print("######------------------------------------Resetting environment...------------------------------------######")


###############################################
# Initiating and Loading the trained RL Agent #
###############################################
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

agent = HITLDQNAgent(n_observations, n_actions, device)
# agent.set_learning(False) # No more random policy exploration     # (comment from Jetwei) I commented this out since 
                                                                    # its giving me the error: AttributeError: 'HITLDQNAgent' 
                                                                    # object has no attribute 'set_learning'

agent.load_model(LOAD_FILEPATH)


##################################################################################
# Evaluating the trained RL Agent by Plotting Cumulative Reward against Episodes #
##################################################################################
#TODO: Update this for all models evaluated
hitl_energy_rewards = evaluate_model(agent, TEST_EPISODES)

plt.figure(figsize=(10, 5))
plt.plot(hitl_energy_rewards, label='hitl+energy')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Comparison (Using Average Reward)')
plt.legend()

display.display(plt.gcf())

plt.ioff()
plt.show()