import torch
import numpy as np

from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase

import pygame

from gym_examples.constants import *
from gym_examples.thermal_comfort_model_sim import ThermalComfortModelSim
from gym_examples.agent_group import AgentGroup
from gym_examples.building import Building
from gym_examples.population_sim import PopulationSim

from pygame.locals import *


"""
References: 
https://github.com/viktor-ktorvi/torch_rl_experimenting/blob/master/custom_env_linear_system.ipynb
https://pytorch.org/tutorials/advanced/coding_ddpg.html
"""
class AirconEnvironment(EnvBase):
    """
    A reinforcement learning environment simulating an air conditioning system in a building.

    Attributes:
        dtype (np.dtype): Data type for numpy arrays.
        is_render (bool): Flag to enable or disable rendering.
        alpha (float): Penalty coefficient for thermal comfort.
        beta (float): Coefficient for another penalty (e.g., energy usage).
        num_days (int): Number of days for the simulation.
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
    def __init__(self, num_days=None, is_render=True, alpha=1, beta=1):
        super(AirconEnvironment, self).__init__()
        self.dtype = np.float32
        self.is_render = is_render
        self.alpha = alpha
        self.beta = beta
        self.num_days = num_days
        self.ambient_temp = -1
        self.vote_up = 0
        self.vote_down = 0
        self.is_terminate = False
        self.temp_setpt = None
        self.reward = 0

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
        self.curr_time = None
        self.thermal_comfort_model = ThermalComfortModelSim() #TODO: Make more complicated
        self.popSim_is_debug = False
        self.popSim = PopulationSim(sample_size=self.sample_size, time_interval=self.time_interval, is_debug=self.popSim_is_debug)
        self.agents_out = self.create_agents()
        self.state = self.get_states()

        self.clock = pygame.time.Clock()
        assert (self.num_days is None) or (self.num_days > 0), "Error: num_days must be either none or greater than 0."

        self.state_size = len(self.get_states())
        self.action_size = 1
        self.action_spec = BoundedTensorSpec(low=20, high=40, shape=torch.Size([self.action_size]))
        observation_spec = UnboundedContinuousTensorSpec(shape=torch.Size([self.state_size]))
        self.observation_spec = CompositeSpec(observation=observation_spec)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size([1]))    

    def _reset(self, tensordict, **kwargs):
        """
        Reset the environment to its initial state.

        Args:
            tensordict (TensorDict): Tensor dictionary with environment data.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorDict: Tensor dictionary with the initial observation.
        """
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
        self.curr_time = None
        self.thermal_comfort_model = ThermalComfortModelSim()
        self.is_terminate = False
        self.ambient_temp = FIXED_POLICY
        self.temp_setpt = self.ambient_temp

        out_tensordict = TensorDict({}, batch_size=torch.Size())
        out_tensordict.set("observation", torch.tensor(self.get_states().flatten(), device=self.device))

        return out_tensordict

    def _step(self, tensordict):
        """
        Execute a step in the environment.

        Args:
            tensordict (TensorDict): Tensor dictionary with the action.

        Returns:
            TensorDict: Tensor dictionary with the next observation, reward, and done flag.
        """
        action = tensordict["action"].cpu().numpy().reshape(-1)[0]
        self.temp_setpt = action

        self.updatePopSim()
        self.curr_time = self.popSim.curr_time
        self.updateTemp() # TODO: Improve the complexity of this

        if self.is_render:
            self.checkEvents()
            self.render()

        if self.num_days is not None and self.curr_time >= timedelta(hours=(self.num_days - 1) * 24 + DAY_END_TIME):
            self.is_terminate = True
            pygame.quit()

        # TODO: To improve
        reward = self.get_reward(action)
        self.reward = reward

        out_tensordict = TensorDict({
            "observation": torch.tensor(self.get_states().flatten(), device=self.device),
            "reward": torch.tensor(reward.astype(np.float32), device=self.device),
            "done": self.is_terminate
        }, batch_size=torch.Size())

        self.state = self.get_states() #TODO: Better way to update states
        return out_tensordict

    def _set_seed(self, seed):
        pass

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

    def updatePopSim(self):
        """
        Update the population simulator, moving agents in and out of the building.
        """
        actual_in_num = len(self.agents_in)
        move_in_num, move_out_num = self.popSim.update(actual_in_num)
        self.moveAgentsIn(move_in_num)
        self.moveAgentsOut(move_out_num)
        
        if self.popSim_is_debug:
            print(" curr_in:", len(self.agents_in), 
                  "\n curr_out:", len(self.agents_out),
                  "\n total:", len(self.agents_in) + len(self.agents_out))
    
    
    def get_states(self):
        """
        Get the current state of the environment.
            # Ambient Temperature (ambient_temp)
            # Number of Up Votes (vote_up)
            # Number of Down Votes (vote_down)
            # Number of Users (# of users)
            # Current Time (curr time)
            # Current Temperature (curr temp)
            # Returns the current state of the simulation, including ambient temperature, votes, number of users, current time, and current temperature
        Returns:
            np.ndarray: Current state of the environment.
        """
        STATE_NUM = 15-9 # 9 individual differences + ambient_temp # vote up + # vote down + # of users + curr time + curr temp
        state = np.zeros(STATE_NUM, dtype=self.dtype)
        if len(self.agents_in) > 0:
            df = AgentGroup(agents=self.agents_in, model=self.thermal_comfort_model).get_group_data_df()
            
            votes = self.thermal_comfort_model.predict_vote(df['user'].values, self.temp_setpt)
            vote_up, vote_down = sum(votes), len(votes) - sum(votes)

            self.vote_up = vote_up
            self.vote_down = vote_down
            state[-6] = self.ambient_temp
            state[-5] = vote_up
            state[-4] = vote_down
            state[-3] = len(df)
            state[-2] = self.curr_time.seconds
            state[-1] = self.temp_setpt
            return state
        else:
            return state
    
    #Handles Pygame events, specifically checking for a quit event to terminate the simulation.
    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit ()
    
    #Sets up the background surface for the Pygame display and fills it with a background color
    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BG_COLOR)

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
                     str(self.temp_setpt) + " / " + str(self.reward), 
                     (280, 7*FONTSIZE))
        pygame.display.update()
    
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

# Checking process
if __name__ == "__main__":
    from torchrl.envs.utils import check_env_specs

    env = AirconEnvironment(30, True, 1, 1) # create environment object
    check_env_specs(env)

    # Simulate applying a constant action 24
    rollout = env.rollout(10, policy=lambda _: TensorDict({"action": 24.0}, batch_size=torch.Size()))
    print("Observation: ", rollout["observation"])




