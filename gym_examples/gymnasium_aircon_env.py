import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from gym_examples.agent_group import AgentGroup
from gym_examples.thermal_comfort_model_sim import ThermalComfortModelSim
from gym_examples.building import Building
from gym_examples.population_sim import PopulationSim

from gym_examples.constants import *


class AirconEnvironment(gym.Env):
    """
    A reinforcement learning environment simulating an air conditioning system in a building.

    Attributes:
        dtype (np.dtype): Data type for numpy arrays.
        is_render (bool): Flag to enable or disable rendering.
        alpha (float): Penalty coefficient for thermal comfort.
        beta (float): Coefficient for another penalty (e.g., energy usage).
        ambient_temp (float): Current ambient temperature.
        vote_up (int): Number of up votes from agents.
        vote_down (int): Number of down votes from agents.
        is_terminate (bool): Flag to indicate if the simulation should terminate.
        temp_setpt (float): Current temperature set point.
        sample_size (int): Number of agents in the simulation.
        time_interval (timedelta): Time interval for updates.
        curr_time (timedelta): Current time in the simulation.
        thermal_comfort_model (ThermalComfortModelSim): Model to simulate thermal comfort.
        popSim_is_debug (bool): Debug flag for the population simulator.
        popSim (PopulationSim): Population simulator.
        agents_in (list): List of agents currently inside the building.
        agents_out (list): List of agents currently outside the building.
        screen (pygame.Surface): Pygame screen surface.
        font (pygame.font.Font): Pygame font.
        clock (pygame.time.Clock): Pygame clock.
        state (np.ndarray): Current state of the environment.
        state_size (int): Size of the state space.
        action_size (int): Size of the action space.
        action_spec (TensorSpec): Specification of the action space.
        observation_spec (CompositeSpec): Specification of the observation space.
        reward_spec (TensorSpec): Specification of the reward space.
    """
    def __init__(self, is_render=True, alpha=1, beta=1):
        # Environment Setup Variables 
        self.is_render = is_render
        self.alpha = alpha
        self.beta = beta

        self.ambient_temp = 0
        self.vote_up = 0
        self.vote_down = 0
        self.temp_setpt = None

        if is_render:
            pygame.init()
            pygame.display.set_caption('HitL AC Simulation')
            self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
            self.font = pygame.font.SysFont("Arial", FONTSIZE)
            self.setBackground()
        else:
            self.screen = None
            self.font = None

        self.building = Building()
        self.agents_in = []
        self.agents_out = []
        self.sample_size = SAMPLE_SIZE
        self.time_interval = TIME_INTERVAL
        self.thermal_comfort_model = ThermalComfortModelSim() #TODO: Make more complicated
        self.popSim_is_debug = False
        self.popSim = PopulationSim(sample_size=self.sample_size, time_interval=self.time_interval, is_debug=self.popSim_is_debug)
        self.agents_out = self.create_agents()
        self.curr_time = DAY_START_TIME

        self.updatePopSim()
        self.curr_time = self.popSim.curr_time
        self.updateTemp() # TODO: Improve the complexity of this

        # self.state_space = spaces.Dict({})
        # self.observation_space = spaces.Dict(
        #     {
        #         "ambient_temp": spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32),
        #         "vote_up": spaces.Box(low=-2**63, high=2**63-1, shape=(1,), dtype=np.float32),
        #         "vote_down": spaces.Box(low=-2**63, high=2**63-1, shape=(1,), dtype=np.float32),
        #         "population_size": spaces.Box(low=-2**63, high=2**63-1, shape=(1,), dtype=np.float32),
        #         "curr_time_sec": spaces.Box(low=-2**63, high=2**63-1, shape=(1,), dtype=np.float32),
        #         "temp_setpt": spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32),
        #     }
        # )
        self.observation_space = spaces.Box(
            low=np.array([0.0, -2**63, -2**63, -2**63, -2**63, 0.0], dtype=np.float32),
            high=np.array([50.0, 2**63-1, 2**63-1, 2**63-1, 2**63-1, 50.0], dtype=np.float32),
            dtype=np.float32
        )
        # self.action_space = spaces.Box(low=0, high=50)
        self.action_space = spaces.Discrete(46) #range between $20-28 \degree C$}, with intervals of $0.2 \degree C$

    def _get_obs(self):
        return {
                    "ambient_temp" : np.array([self.ambient_temp], dtype=np.float32),
                    "vote_up" : np.array([self.vote_up], dtype=np.float32),
                    "vote_down" : np.array([self.vote_down], dtype=np.float32),
                    "population_size" : np.array([len(AgentGroup(agents=self.agents_in, model=self.thermal_comfort_model).get_group_data_df())], dtype=np.float32),
                    "curr_time_sec" : np.array([self.curr_time.seconds], dtype=np.float32),
                    "temp_setpt": np.array([self.temp_setpt], dtype=np.float32),
                }

    def _get_info(self):
        """Auxiliary information returned by step and reset"""
        #TODO: 
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset all variables
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
        self.popSim = PopulationSim(sample_size=self.sample_size, time_interval=self.time_interval, is_debug=self.popSim_is_debug)
        self.agents_out = self.create_agents()
        self.curr_time = self.popSim.curr_time
        self.thermal_comfort_model = ThermalComfortModelSim()
        self.is_terminate = False
        self.ambient_temp = 0
        self.temp_setpt = self.ambient_temp

        observation = self._get_obs()
        info = self._get_info()

        if self.is_render:
            self.render()

        return dict_to_np(observation), info

    def step(self, action: int):
        self.temp_setpt = action
        temp_setpt = 20+action*0.2 #TODO: Is this the correct way - convert state to temp_setpt 
        self.updatePopSim()
        self.curr_time = self.popSim.curr_time
        self.updateTemp() # TODO: Improve the complexity of this


        df = AgentGroup(agents=self.agents_in, model=self.thermal_comfort_model).get_group_data_df()
        votes = self.thermal_comfort_model.predict_vote(df['user'].values, self.temp_setpt)
        vote_up, vote_down = sum(votes), len(votes) - sum(votes)
        self.vote_up = vote_up
        self.vote_down = vote_down

        # Prepare observation, reward, terminated, False, info
        observation = self._get_obs()
        reward = self.get_reward(temp_setpt)
        if self.curr_time >= timedelta(hours=DAY_END_TIME):
            terminated = True
        else:
            terminated = False
        info = self._get_info()

        if self.is_render:
            self.render()

        return dict_to_np(observation), reward, terminated, False, info

    def render(self):
        if self.is_render:
            if self.clock is None:
                self.clock = pygame.time.Clock()

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
                        str(self.temp_setpt) + " / " + str(self.reward), 
                        (280, 7*FONTSIZE))
            pygame.display.update()
            return self._render_frame()
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def get_reward(self, temp_setpt):
        """
        Calculate the reward based on the current temperature set point.

        Args:
            temp_setpt (float): Temperature set point.

        Returns:
            float: Calculated reward.
        """
        #TODO: Fix the structure AgentGroup code structure and make it more understandable
        rl_reward, _ = AgentGroup(agents=self.agents_in, 
                                model=self.thermal_comfort_model
                                ).predict(temp=self.temp_setpt, 
                                            ambient_temp=self.ambient_temp,
                                            alpha=self.alpha,
                                            beta=self.beta)        
        return rl_reward

    def updatePopSim(self):
        """
        Update the population simulator, moving agents in and out of the building.
        """
        actual_in_num = len(self.agents_in)
        move_in_num, move_out_num = self.popSim.update(actual_in_num)
        self.moveAgentsIn(move_in_num)
        self.moveAgentsOut(move_out_num)

    def moveAgentsOut(self, num):
        """
        Move agents out of the building.

        Args:
            num (int): Number of agents to move out.
        """
        if num > len(self.agents_in):
            raise ValueError("Attempted to move more agents out than the number of agents_in!")
        # move agents out
        for agent in self.agents_in[:num]:
            agent.randomOut()
        self.agents_out += self.agents_in[:num]
        self.agents_in = self.agents_in[num:]
        
        
    def moveAgentsIn(self, num):
        """
        Move agents out of the building.

        Args:
            num (int): Number of agents to move out.
        """
        if num > len(self.agents_out):
            raise ValueError("Attempted to move more agents in than the number of agents_out!")
        # move agents in
        for agent in self.agents_out[:num]:
            agent.randomIn()
        self.agents_in += self.agents_out[:num]
        self.agents_out = self.agents_out[num:]
        
    def create_agents(self):
        """
        Create a group of agents for the simulation.

        Returns:
            list: List of created agents.
        """
        agent_group = AgentGroup(
            group_size=self.popSim.sample_size,
            font=self.font,
            screen=self.screen,
            model=self.thermal_comfort_model
        )
        return agent_group.generate()

    def updateTemp(self):
        """
        Update the ambient temperature based on the current time.
        """
        if self.curr_time.seconds < 3600 * 12:
            lower = max(self.ambient_temp, 27) # make temperature increase
            upper = 30
        elif self.curr_time.seconds < 3600 * 18:
            lower, upper = 29, 31
        else:
            upper = min(30, self.ambient_temp)
            lower = 27
        self.ambient_temp = round(np.random.uniform(lower, upper), 1)

    def setBackground(self):
        """
        Sets up the background surface for the Pygame display and fills it with a background color
        """
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BG_COLOR)
    
def dict_to_np(input_dict):
    # Extract values from the dictionary and convert to a numpy array
    print(input_dict)
    np_array = np.array(list(input_dict.values()), dtype=np.float32).flatten()
    
    # Convert numpy array to a torch tensor with float32 type
    # torch_tensor = torch.tensor(np_array, dtype=torch.float32)

    return np_array

"""
Verification
cd gym_examples
python gymnasium_aircon_env.py
"""
if __name__ == "__main__":
    #TODO: Test environment
    from stable_baselines3.common.env_checker import check_env

    env = AirconEnvironment(False, 30, 1, 1)
    check_env(env)
    pass