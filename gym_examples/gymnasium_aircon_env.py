import numpy as np
from scipy.optimize import minimize, basinhopping

import gymnasium as gym
from gymnasium import spaces

from gym_examples.utils.building import Building
from gym_examples.utils.population import PopulationSimulation

from gym_examples.utils.constants import *
from gym_examples.utils.simulation_visualiser import SimulationVisualiser


class AirconEnvironment(gym.Env):
    """
    A reinforcement learning environment simulating an air conditioning system in a building. 

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
    def __init__(self, is_render=False, check_optimal=False, w_usercomfort=1, w_energy=1):
        # Environment Setup Variables 
        self.is_render = is_render
        self.w_usercomfort = w_usercomfort
        self.w_energy = w_energy
        self.check_optimal = check_optimal

        self.curr_time = timedelta(days=0, hours=DAY_START_TIME)

        self.ambient_temp = 0
        self.update_ambient_temp() # TODO: Improve the complexity of this
        self.temp_setpt = self.ambient_temp

        self.building = Building()
        self.population_simulation = PopulationSimulation(SAMPLE_SIZE, TIME_INTERVAL)

        self.render_engine = SimulationVisualiser(is_render)


        self.is_terminate = False

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
            low=np.array([-2**63, -2**63, -2**63, -2**63, -2**63, 2**63-1], dtype=np.float32),
            high=np.array([2**63-1, 2**63-1, 2**63-1, 2**63-1, 2**63-1, 2**63-1], dtype=np.float32),
            dtype=np.float32
        )
        # self.action_space = spaces.Box(low=0, high=50)
        self.action_space = spaces.Discrete(46) #range between $20-28 \degree C$}, with intervals of $0.2 \degree C$

    def _get_info(self):
        """Auxiliary information returned by step and reset"""
        if self.check_optimal: 
            optimal_temp = self.optimize_temperature()
            return {"optimal_temp": optimal_temp}
        else:
            return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.curr_time = timedelta(days=0, hours=DAY_START_TIME)

        self.ambient_temp = 0
        self.update_ambient_temp() # TODO: Improve the complexity of this
        self.temp_setpt = self.ambient_temp
        
        self.building = Building()
        self.population_simulation = PopulationSimulation(SAMPLE_SIZE, TIME_INTERVAL)

        self.render_engine = SimulationVisualiser()


        self.is_terminate = False

        observation = self._get_obs()
        info = {} # self._get_info()

        if self.is_render:
            reward = 0
            up_votes, down_votes = self.population_simulation.get_vote_up_count(self.temp_setpt)
            self.render_engine.update(
                            humans_in=self.population_simulation.get_humans(True), 
                            humans_out=self.population_simulation.get_humans(False),
                            building=self.building, 
                            reward=0,
                            curr_time=self.curr_time,
                            vote_up=up_votes,
                            vote_down=down_votes,
                            temp_setpt=self.temp_setpt,
                            time_interval=TIME_INTERVAL,
                            sample_size=SAMPLE_SIZE,
                        )
        return dict_to_np(observation), info

    def step(self, action: int):
        self.curr_time += TIME_INTERVAL

        self.temp_setpt = 20 + action * 0.2 # Convert action to temperature setpoint
        self.update_ambient_temp() # TODO: Improve the complexity of this

        self.population_simulation.step()

        # Prepare observation, reward, terminated, False, info
        observation = self._get_obs()
        reward = self.get_reward(self.temp_setpt)
        terminated = True if self.curr_time >= timedelta(hours=DAY_END_TIME) else False
        info = self._get_info()
        up_votes, down_votes = self.population_simulation.get_vote_up_count(self.temp_setpt)

        # Update visualiser
        if self.is_render and not terminated:
            print("Updating display....")
            self.render_engine.update(
                                        humans_in=self.population_simulation.get_humans(True), 
                                        humans_out=self.population_simulation.get_humans(False),
                                        building=self.building, 
                                        reward=reward,
                                        curr_time=self.curr_time,
                                        vote_up=up_votes,
                                        vote_down=down_votes,
                                        temp_setpt=self.temp_setpt,
                                        time_interval=TIME_INTERVAL,
                                        sample_size=SAMPLE_SIZE,
                                    )
        elif self.is_render and terminated:
            print("Quitting visualiser...")
            self.render_engine.close()
        print(f'{observation}')
        print(f'tempsetpt:{self.temp_setpt}, reward:{reward}')
        return dict_to_np(observation), reward, terminated, False, info

    def _get_obs(self):
        up_votes, down_votes = self.population_simulation.get_vote_up_count(self.temp_setpt) if self.temp_setpt!= None else (0,0)
        return {
                    "ambient_temp" : np.array([self.ambient_temp], dtype=np.float32),
                    "temp_setpt": np.array([self.temp_setpt], dtype=np.float32),
                    "vote_up" : np.array([up_votes], dtype=np.float32),
                    "vote_down" : np.array([down_votes], dtype=np.float32),
                    "population_size" : np.array([len(self.population_simulation.get_humans(True))], dtype=np.float32),
                    "curr_time_sec" : np.array([self.curr_time.seconds], dtype=np.float32),
                }
    
    def get_reward(self, temp):
        """
        Calculate the true reward based on the current temperature set point.

        Args:
            temp (float): Temperature set point.

        Returns:
            float: Calculated reward.
        """
        # def get_reward(self, temp, ambient_temp, w_usercomfort=1, w_energy=0.03):
        # Energy Usage
        q = -m * c_p * abs(temp - self.ambient_temp)
        power = q / (TIME_INTERVAL / timedelta(hours=1)) / 1000
        # User 
        # print(f'Ambient:{self.ambient_temp}, Set:{temp}')
        up_votes, down_votes = self.population_simulation.get_vote_up_count(temp)
        n_humans_in = len(self.population_simulation.get_humans(True))
        # print(f'Upvotes:{up_votes}, Downvotes:{down_votes} Number of humans {n_humans_in}')

        #NOTE: Doesnt work when there are many votes but all have discomfort vote of 1 
        # average_user_comfort_vote = -(up_votes+down_votes)/n_humans_in 
        average_user_comfort_vote = -(up_votes+down_votes) 

        # print("Comfort score:", average_user_comfort_vote, "Power score:", power)
        reward = (self.w_usercomfort * average_user_comfort_vote) +self.w_energy*power
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
        def objective(temp):
            return -self.get_reward(temp)
        
        # Perform the optimization
        result1 = minimize(objective, 20, method='Nelder-Mead', bounds=[(self.ambient_temp-15, self.ambient_temp+15)])
        result2 = minimize(objective, 35, method='BFGS')

        max_reward1 = self.get_reward(result1.x[0])
        max_reward2 = self.get_reward(result2.x[0])

        if max_reward1>max_reward2: result = result1
        else: result = result2
        print(f'optimal_temp:{result.x[0]}, reward:{self.get_reward(result.x[0])}')
        # minimizer_kwargs = {"method": "BFGS"}
        # result = basinhopping(objective, 25, minimizer_kwargs=minimizer_kwargs, niter=200)
        return result.x[0]
    
def dict_to_np(input_dict):
    # Extract values from the dictionary and convert to a numpy array
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