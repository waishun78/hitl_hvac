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

class HITLAirconEnvironment(gym.Env):
    """
    A reinforcement learning environment simulating an air conditioning system in a building with rewards derived only from user comfort. 
    Using the MDP version where met and clo is an observable for the agent

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
        self.prev_pmv = [0 for _ in range(7)]

        # self.render_engine = SimulationVisualiser(is_render)

        self.is_terminate = False

        low = np.full((3 + HITLAirconEnvironment.MAX_POP_SIZE*2, 1), 0, dtype=np.float32)
        low[0] = -273 #ambient temp
        low[1] = 20 # temp_setpt
        low[2] = -2**63 # curr_time_sec
        low[3:3+HITLAirconEnvironment.MAX_POP_SIZE] = 1 # met lower bound is 1

        high = np.full((3 + HITLAirconEnvironment.MAX_POP_SIZE*2, 1), 0, dtype=np.float32)
        high[0] = 50
        high[1] = 28
        high[2] = 2**63-1
        high[3:3+HITLAirconEnvironment.MAX_POP_SIZE] = 4 # met upper bound is 1
        high[4+HITLAirconEnvironment.MAX_POP_SIZE:] = 1.5 # clo upper bound is 1

        # self.state_space = spaces.Dict({})
        self.observation_space = spaces.Box(
            low=low,
            high=high
        )
        # self.action_space = spaces.Box(low=0, high=50)
        self.action_space = spaces.Discrete(46) #range between $20-28 \degree C$}, with intervals of $0.2 \degree C$

    def _get_info(self) -> Dict:
        """Auxiliary information returned by step and reset"""
        # if self.check_optimal: 
        #     optimal_temp = self.optimize_temperature()
        #     return {"optimal_temp": optimal_temp}
        # else:
        #     return {}
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

        # if self.is_render:
        #     reward = 0
        #     up_votes, down_votes = self.population_simulation.get_cum_pmv(self.temp_setpt)
        #     self.render_engine.update(
        #                     humans_in=self.population_simulation.get_df(), 
        #                     humans_out=self.population_simulation.get_humans(False),
        #                     building=self.building, 
        #                     reward=0,
        #                     curr_time=self.curr_time,
        #                     vote_up=up_votes,
        #                     vote_down=down_votes,
        #                     temp_setpt=self.temp_setpt,
        #                     time_interval=TIME_INTERVAL,
        #                     sample_size=SAMPLE_SIZE,
        #                 )
        return observation, info

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

        return observation, reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        # pmv = self.population_simulation.get_pmv(self.temp_setpt) if self.temp_setpt!= None else (0,0)
        df = self.population_simulation.get_comfort_profile_df()
        obs = np.empty((3+HITLAirconEnvironment.MAX_POP_SIZE*2))
        obs[:] = np.nan
        obs[0] = self.ambient_temp
        obs[1] = self.temp_setpt
        obs[2] = self.curr_time.seconds
        obs[3:3+HITLAirconEnvironment.MAX_POP_SIZE] = df.loc[:,"met"]
        obs[3+HITLAirconEnvironment.MAX_POP_SIZE:] = df.loc[:, "clo"]
        # torch_tensor = torch.tensor(obs, dtype=torch.float32)

        return obs
    
    def get_reward(self, temp: int) -> int:
        """
        Calculate the true reward based on the current temperature set point.

        Args:
            temp (float): Temperature set point.

        Returns:
            float: Calculated reward.
        """

        pmv_dist = self.population_simulation.get_pmv(temp)
        # print(f'Upvotes:{up_votes}, Downvotes:{down_votes} Number of humans {n_humans_in}')

        average_user_comfort_vote = 0

        cum_pmv, prev_cum_pmv = 0, 0
        for i in range(3):
            cum_pmv += (pmv_dist[i] + pmv_dist[len(pmv_dist)-1-i])*(3-i) # -3/+3 PMV means 3 votes
            prev_cum_pmv += (self.prev_pmv[i] + self.prev_pmv[len(self.prev_pmv)-1-i])*(3-i)

        if sum(pmv_dist) > 0: pmv_f = cum_pmv/(sum(pmv_dist)) # if no humans in building
        else: pmv_f = 0
        if sum(self.prev_pmv) > 0: ppmv_f = prev_cum_pmv/(sum(self.prev_pmv))
        else: ppmv_f = 0

        average_user_comfort_vote = pmv_f - self.discount * ppmv_f
        # average_user_comfort_vote = average_user_comfort_vote #TODO: Random Tunable scale
        # print(f'PMV:{pmv_f}, {ppmv_f},{average_user_comfort_vote}')

        self.prev_pmv = pmv_dist

        reward = (self.w_usercomfort * average_user_comfort_vote)

        # Store for info
        self.comfort_score = self.w_usercomfort * average_user_comfort_vote

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
        # def objective(temp):
        #     return -self.get_reward(temp)
        
        # # Perform the optimization
        # result1 = minimize(objective, 20, method='Nelder-Mead', bounds=[(self.ambient_temp-15, self.ambient_temp+15)])
        # result2 = minimize(objective, 30, method='Nelder-Mead', bounds=[(self.ambient_temp-15, self.ambient_temp+15)])

        # max_reward1 = self.get_reward(result1.x[0])
        # max_reward2 = self.get_reward(result2.x[0])

        # if max_reward1>max_reward2: result = result1
        # else: result = result2
        # print(f'optimal_temp:{result.x[0]}, reward:{self.get_reward(result.x[0])}')
        # # minimizer_kwargs = {"method": "BFGS"}
        # # result = basinhopping(objective, 25, minimizer_kwargs=minimizer_kwargs, niter=200)
        # return result.x[0]
    
# def dict_to_np(input_dict) -> np.ndarray:
#     # Extract values from the dictionary and convert to a numpy array
#     np_array = np.array(list(input_dict.values()), dtype=np.float32).flatten()
    
#     # Convert numpy array to a torch tensor with float32 type
#     # torch_tensor = torch.tensor(np_array, dtype=torch.float32)

#     return np_array

"""
Verification
cd gym_examples
python gymnasium_aircon_env.py
"""
if __name__ == "__main__":
    #TODO: Test environment
    from stable_baselines3.common.env_checker import check_env

    env = HITLAirconEnvironment(False, 30, 1, 1)
    check_env(env)
    pass