import pygame
from pygame.locals import *
from datetime import timedelta
import numpy as np
import pandas as pd
import pickle
import time

from constants import *
from population_sim import PopulationSim
from building import Building
from agent_group import AgentGroup
from thermal_comfort_model_sim import ThermalComfortModelSim

#The code implements a simulation environment for a Human-in-the-Loop Air Conditioning (HitL AC) system. It simulates the thermal comfort of a building's occupants and allows an RL agent to control the temperature. The simulation includes a population of agents, a thermal comfort model, and a building environment. The goal is to optimize the temperature setting to maximize the comfort of the occupants while potentially considering energy efficiency.
class Simulation(object):
    def __init__(self, num_days=None, is_render=True, alpha=1, beta=1):
        if is_render:
            pygame.init()
            pygame.display.set_caption('HitL AC Simulation')
            self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
            self.font = pygame.font.SysFont("Arial", FONTSIZE)
        else:
            self.screen = None
            self.font = None
        self.clock = pygame.time.Clock()
        self.num_days = num_days
        assert (self.num_days is None) or (self.num_days > 0), "Error: num_days must be either none or greater than 0."
        self.is_render = is_render

        self.agents_in = []
        self.agents_out = []
        
        self.popSim = None
        self.sample_size = SAMPLE_SIZE
        self.time_interval = TIME_INTERVAL
        self.popSim_is_debug = False
        
        self.curr_time = None
        self.thermal_comfort_model = ThermalComfortModelSim()# pickle.load(open(THERMAL_COMFORT_MODEL, "rb"))
        
        self.rl_policy_rewards = []
        self.reward = 0
        self.rl_temp = -1 # initial temp
        self.fixed_temp = FIXED_POLICY
        self.fixed_policy_rewards = []
        self.fixed_reward = 0

        self.is_terminate = False
        self.ambient_temp = -1
        self.alpha = alpha # 0 means no penalty for thermal comfort, 1 is recommended
        self.beta = beta
        self.vote_up = 0
        self.vote_down = 0
        
        
    #Sets up the background surface for the Pygame display and fills it with a background color
    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BG_COLOR)

    #Moves a specified number of agents from inside to outside the building. If the number exceeds the current number of agents inside, it raises an error
    def moveAgentsOut(self, num):
        if num > len(self.agents_in):
            raise ValueError("Attempted to move more agents out than the number of agents_in!")
        # move agents out
        for agent in self.agents_in[:num]:
            agent.randomOut()
        self.agents_out += self.agents_in[:num]
        self.agents_in = self.agents_in[num:]
        
        
     #Moves a specified number of agents from outside to inside the building. If the number exceeds the current number of agents outside, it raises an error       
    def moveAgentsIn(self, num):
        if num > len(self.agents_out):
            raise ValueError("Attempted to move more agents in than the number of agents_out!")
        # move agents in
        for agent in self.agents_out[:num]:
            agent.randomIn()
        self.agents_in += self.agents_out[:num]
        self.agents_out = self.agents_out[num:]
          
    #Initializes the population simulator and agent group, generating agents and setting up the building environment.
    def startSimulation(self):
        if self.is_render:
            self.setBackground()
        self.popSim = PopulationSim(sample_size=self.sample_size,
                                    time_interval=self.time_interval, 
                                    is_debug=self.popSim_is_debug)
        agent_group = AgentGroup(group_size=self.popSim.sample_size, 
                                 font=self.font,
                                 screen=self.screen,
                                 model=self.thermal_comfort_model)
        self.agents_out = agent_group.generate()
        #print("User modes: \n", agent_group.get_group_data_df().value_counts())

        self.building = Building()

    def reset(self):
        if self.is_render:
            pygame.init()
            pygame.display.set_caption('HitL AC Simulation')
            self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
            self.font = pygame.font.SysFont("Arial", FONTSIZE)
        else:
            self.screen = None
            self.font = None
        self.clock = pygame.time.Clock()
        self.agents_in = []
        self.agents_out = []
        self.popSim = None
        self.curr_time = None
        self.reward = 0
        self.rl_policy_rewards = []
        self.fixed_policy_rewards = []

        self.is_terminate = False
        self.startSimulation()
        self.ambient_temp = FIXED_POLICY
        return self.get_states()
    # State variables
    #Ambient Temperature (ambient_temp)
    #Number of Up Votes (vote_up)
    #Number of Down Votes (vote_down)
    #Number of Users (# of users)
    #Current Time (curr time)
    #Current Temperature (curr temp)
    #Returns the current state of the simulation, including ambient temperature, votes, number of users, current time, and current temperature
    def get_states(self):
        STATE_NUM = 15-9 # 9 individual differences + ambient_temp # vote up + # vote down + # of users + curr time + curr temp
        state = np.zeros(STATE_NUM)
        if len(self.agents_in) > 0:
            df = AgentGroup(agents=self.agents_in, model=self.thermal_comfort_model).get_group_data_df()
            #ct = df['user'].value_counts() 
            #ct = ct / sum(ct)
            #for i in range(9):
            #    state[i] = ct[i] if i in ct.index else 0
            
            votes = self.thermal_comfort_model.predict_vote(df['user'].values, self.rl_temp)
            vote_up, vote_down = sum(votes), len(votes) - sum(votes)

            
            self.vote_up = vote_up
            self.vote_down = vote_down
            state[-6] = self.ambient_temp
            state[-5] = vote_up
            state[-4] = vote_down
            state[-3] = len(df)
            state[-2] = self.curr_time.seconds
            state[-1] = self.rl_temp
            return state
        else:
            return state


#The reward function aims to balance the thermal comfort of the agents and the energy consumption of the air conditioning system
# THis needs to modified
#Calculates and returns the rewards for both the RL policy and the fixed policy based on the thermal comfort model
    def get_reward(self):
        if len(self.agents_in) > 0:
            rl_reward, rl_pmv_mean = AgentGroup(agents=self.agents_in, 
                                                model=self.thermal_comfort_model
                                                ).predict(temp=self.rl_temp, 
                                                          ambient_temp=self.ambient_temp,
                                                          alpha=self.alpha,
                                                          beta=self.beta)
            fp_reward, fp_pmv_mean = AgentGroup(agents=self.agents_in, 
                                                model=self.thermal_comfort_model
                                                ).predict(temp=self.fixed_temp, 
                                                          ambient_temp=self.ambient_temp,
                                                          alpha=self.alpha,
                                                          beta=self.beta)
            
            self.rl_policy_rewards.append(rl_pmv_mean)
            self.fixed_policy_rewards.append(fp_pmv_mean)
            
            self.reward = rl_reward#self.rl_policy_rewards[-1]#_calc_nanmean(self.rl_policy_rewards)
            self.fixed_reward = fp_reward#self.fixed_policy_rewards[-1]#_calc_nanmean(self.fixed_policy_rewards)
            # print(f"RL Reward: {rl_reward}")
            # print(f"Fixed Reward: {fp_reward}")
            # print('='*20)
        #return np.sum(self.rl_policy_rewards), np.sum(self.fixed_policy_rewards)
        return self.reward, self.fixed_reward
    
    
    #Returns the cumulative rewards for both the RL policy and the fixed policy, skipping the first element
    def get_cummulative_reward(self):
        return np.sum(self.rl_policy_rewards[1:]), np.sum(self.fixed_policy_rewards[1:])

#Advances the simulation by one step with the given temperature setting, updates the state, and returns the new state, reward, and additional info
    def step(self, temp):
        self.rl_temp = temp
        #reward, _ = self.reward #self.get_reward()
        self.update()
        observation = self.get_states()
        
        return observation, self.reward, (self.is_terminate, self.ambient_temp, len(self.agents_in))

#Updates the population simulator, moving agents in and out of the
    def updatePopSim(self):
        actual_in_num = len(self.agents_in)
        move_in_num, move_out_num = self.popSim.update(actual_in_num)
        self.moveAgentsIn(move_in_num)
        self.moveAgentsOut(move_out_num)
        
        if self.popSim_is_debug:
            print(" curr_in:", len(self.agents_in), 
                  "\n curr_out:", len(self.agents_out),
                  "\n total:", len(self.agents_in) + len(self.agents_out))

#Updates the ambient temperature based on the current time, simulating daily temperature variations
    def updateTemp(self):
        if self.curr_time.seconds < 3600 * 12:
            lower = max(self.ambient_temp, 27) # make temperature increase
            upper = 30
        elif self.curr_time.seconds < 3600 * 18:
            lower, upper = 29, 31
        else:
            upper = min(30, self.ambient_temp)
            lower = 27
        self.ambient_temp = round(np.random.uniform(lower, upper), 1)


# Updates the population simulation, time, rewards, and temperature. If rendering is enabled, it checks for events and renders the simulation. It also checks for simulation termination conditions
    def update(self):
        #time.sleep(3)
        self.updatePopSim()
        self.curr_time = self.popSim.curr_time
        self.get_reward()
        self.updateTemp()

        if self.is_render:
            self.checkEvents()
            self.render()
        
        if self.num_days is not None:
            if self.curr_time >= timedelta(hours=(self.num_days - 1)*24 + DAY_END_TIME):
                self.is_terminate = True
                pygame.quit()

#Handles Pygame events, specifically checking for a quit event to terminate the simulation.
    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit ()


#Renders the simulation environment, including the building and agents, and displays various statistics on the screen
    def render(self):
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.background, (0, 0))
        self.building.render(self.screen)
        for agent in self.agents_in:
            agent.render()
        for agent in self.agents_out:
            agent.render()

        # display stats
        def _displayText(description, pos_description, value, pos_value):
            text_description = self.font.render(description, True, BLACK)
            self.screen.blit(text_description, text_description.get_rect(center=pos_description))
            text_value = self.font.render(value, True, BLACK)
            self.screen.blit(text_value, text_value.get_rect(center=pos_value))

        _displayText("Total population of the day: ", 
                     (75, FONTSIZE),
                     str(self.sample_size), 
                     (165, FONTSIZE))
        _displayText("Time: ", 
                     (17, 3*FONTSIZE),
                     f"{self.curr_time - self.time_interval} ~ {self.curr_time}", 
                     (145, 3*FONTSIZE))
        _displayText("Votes (UP / DOWN): ", 
                     (57, 5*FONTSIZE),
                     str(self.vote_up) + " / " + str(self.vote_down), 
                     (140, 5*FONTSIZE))
        _displayText("HLAC RL (temperature / reward): ", 
                     (90, 7*FONTSIZE),
                     str(self.rl_temp) + " / " + str(self.reward), 
                     (280, 7*FONTSIZE))
        _displayText("Set point (temperature / reward): ", 
                     (90, 9*FONTSIZE),
                     str(self.fixed_temp) + " / " + str(self.fixed_reward), 
                     (280, 9*FONTSIZE))
        pygame.display.update()