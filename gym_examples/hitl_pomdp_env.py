from enum import Enum
from typing import Dict
import numpy as np
from scipy.optimize import minimize

import gymnasium as gym
from gymnasium import spaces
import torch

from gym_examples.utils.building import Building
from gym_examples.utils.population import PopulationSimulation

from gym_examples.utils.constants import *
from gym_examples.utils.simulation_visualiser import SimulationVisualiser
from gym_examples.utils.thermal_comfort_model_sim import ThermalComfortModelSim

class HITLPOMDPAirconEnvironment(gym.Env):
    """
    A reinforcement learning environment simulating an air conditioning system in a building with rewards derived only from user comfort. 
    Using the POMDP version where only votes are observable for the agent

    Attributes:
        is_render (bool): Flag to enable or disable rendering.
        w_usercomfort (float): Penalty coefficient for thermal comfort.
        w_energy (float): Coefficient for another penalty (e.g., energy usage).
        ambient_temp (float): Current ambient temperature.
        temp_setpt (float): Current temperature set point.
        curr_time (timedelta): Current time in the simulation.
        state: Current state of the environment.
        action_space: Specification of the action space.
        observation_space (CompositeSpec): Specification of the observation space.
    """
    MAX_POP_SIZE = 600
    def __init__(self, population_simulation:PopulationSimulation, is_render=False, check_optimal=False, w_usercomfort=1, discount=0.8):
        # Environment Setup Variables 
        self.is_render = is_render
        self.w_usercomfort = w_usercomfort
        self.check_optimal = check_optimal
        self.discount = discount

        self.curr_time = timedelta(days=0, hours=DAY_START_TIME)

        self.ambient_temp = 0
        self.update_ambient_temp() # TODO: Improve the complexity of this
        self.temp_setpt = self.ambient_temp

        self.building = Building()
        self.population_simulation = population_simulation
        # self.prev_pmvf = [1 for _ in range(9)] # NOTE: If all zero, it will be in a perfect state. Init state gives equal votes to all states
        self.prev_pmvf = 0

        self.is_terminate = False

        self.observation_space = spaces.Box(
            low=np.array([
                -2**63, # ambient temp
                -2**63, # temp_setpt
                0, # PMV -4
                0, # PMV -3
                0, # PMV -2
                0, # PMV -1
                0, # PMV  0
                0, # PMV +1
                0, # PMV +2
                0, # PMV +3
                0, # PMV +4
                -2**63, # curr_time_sec
                ], dtype=np.float32),
            high=np.array([
                2**63-1, 
                2**63-1, 
                2**63-1, 
                2**63-1, 
                2**63-1,
                2**63-1, 
                2**63-1, 
                2**63-1, 
                2**63-1, 
                2**63-1, 
                2**63-1, 
                2**63-1, 
                ], dtype=np.float32),
            dtype=np.float32
        )
        # self.action_space = spaces.Box(low=0, high=50)
        self.action_space = spaces.Discrete(46) #range between $20-28 \degree C$}, with intervals of $0.2 \degree C$
        self.weights = np.array([i for i in range(ThermalComfortModelSim.max_pmv, 0, -1)] + [i for i in range(ThermalComfortModelSim.max_pmv + 1)])


    def _get_info(self) -> Dict:
        """Auxiliary information returned by step and reset"""
        return {}

    def reset(self, seed=None, options=None) -> None:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.curr_time = timedelta(days=0, hours=DAY_START_TIME)

        self.ambient_temp = 0
        self.update_ambient_temp() # TODO: Improve the complexity of this
        self.temp_setpt = self.ambient_temp
        
        self.building = Building()
        self.population_simulation.reset()

        self.render_engine = SimulationVisualiser()


        self.is_terminate = False

        observation = self._get_obs()
        info = self._get_info()

        return dict_to_np(observation), info

    def step(self, action: int) -> np.ndarray:
        self.curr_time += TIME_INTERVAL

        self.temp_setpt = 20 + action * 0.2 # Convert action to temperature setpoint
        self.update_ambient_temp() # TODO: Improve the complexity of this

        self.population_simulation.step()

        # Prepare observation, reward, terminated, False, info
        observation = self._get_obs()
        reward = self.get_reward(self.temp_setpt)
        terminated = True if self.curr_time >= timedelta(hours=DAY_END_TIME) else False
        info = self._get_info()

        return dict_to_np(observation), reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        pmv = self.population_simulation.get_observed_pmv(self.temp_setpt) if self.temp_setpt!= None else (0,0)
        return {
                    "ambient_temp" : np.array([self.ambient_temp], dtype=np.float32),
                    "temp_setpt": np.array([self.temp_setpt], dtype=np.float32),
                    "pmv_n4": np.array([pmv[0]], dtype=np.float32),
                    "pmv_n3": np.array([pmv[1]], dtype=np.float32),
                    "pmv_n2": np.array([pmv[2]], dtype=np.float32),
                    "pmv_n1": np.array([pmv[3]], dtype=np.float32),
                    "pmv_0": np.array([pmv[4]], dtype=np.float32),
                    "pmv_p1": np.array([pmv[5]], dtype=np.float32),
                    "pmv_p2": np.array([pmv[6]], dtype=np.float32),                   
                    "pmv_p3": np.array([pmv[7]], dtype=np.float32),                   
                    "pmv_p4": np.array([pmv[8]], dtype=np.float32),                   
                    "curr_time_sec" : np.array([self.curr_time.seconds], dtype=np.float32),
                }
    
    def get_reward(self, temp: int) -> int:
        """
        Calculate the true reward based on the current temperature set point.

        Args:
            temp (float): Temperature set point.

        Returns:
            float: Calculated reward.
        """
        pmv_dist = self.population_simulation.get_true_pmv(temp)
        # Calculate cumulative PMV for current and previous distributions
        cum_pmv = np.sum(pmv_dist * self.weights)
        pmv_sum = np.sum(pmv_dist)
        pmv_f = cum_pmv / pmv_sum if pmv_sum > 0 else 0
        #TODO: Tune self.discount hyperparameter (higher discount, less emphasis given to state change, more emphasis given to current state change)
        average_user_comfort_vote = pmv_f - self.discount * self.prev_pmvf
        reward = self.w_usercomfort * average_user_comfort_vote

        self.comfort_score = self.w_usercomfort * average_user_comfort_vote
        self.prev_pmvf = pmv_f

        return reward

    def update_ambient_temp(self):
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

    def optimize_temperature(self): #TODO: Make this more accurate - some setpt temperatures have better rewards
        """Find the optimal temperature to use for the system"""
        # Define the objective function to minimize (negative of reward function for maximization)
        pass
    
def dict_to_np(input_dict) -> np.ndarray:
    # Extract values from the dictionary and convert to a numpy array
    np_array = np.array(list(input_dict.values()), dtype=np.float32).flatten()
    return np_array

"""
Verification
cd gym_examples
python gymnasium_aircon_env.py
"""
if __name__ == "__main__":
    #TODO: Test environment
    from stable_baselines3.common.env_checker import check_env

    # env = HITLPOMDPAirconEnvironment(False, 30, 1, 1)
    # check_env(env)
    pass