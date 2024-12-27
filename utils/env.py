import gym
from gym import spaces
import numpy as np
import random
from torch import manual_seed
from pythermalcomfort.models import pmv_ppd

"""
A custom Gym environment for simulating a building's HVAC system and user interactions.

Attributes:
    metadata (dict): Metadata for rendering modes.
    alpha (float): Weight for comfort penalty in reward calculation.
    beta (float): Weight for energy penalty in reward calculation.
    EER (float): Energy Efficiency Ratio for HVAC power calculation.
    seed_value (int): Seed for random number generation.
    rng (RandomState): Random number generator instance.
    room_width (int): Width of the room in the simulation.
    room_height (int): Height of the room in the simulation.
    action_space (Discrete): Action space for the environment.
    observation_space (Box): Observation space for the environment.
    adversary_action_space (Discrete): Action space for adversary actions.
    time_step (int): Current time step in the simulation.
    max_steps (int): Maximum number of steps per episode.
    indoor_temp (float): Current indoor temperature.
    setpoint_temp (float): Desired indoor temperature.
    outdoor_temp (float): Current outdoor temperature.
    occupancy_level (float): Current occupancy level.
    num_users (int): Number of users in the building.
    users (list): List of user attributes.
    user_feedback (float): Aggregated user feedback on comfort.
    hvac_power (float): Current power consumption of the HVAC system.
    history (dict): Historical data of the simulation.

Methods:
    generate_outdoor_temperature(): Generates the outdoor temperature based on time of day.
    generate_occupancy_level(): Generates the occupancy level based on time of day.
    generate_num_users(): Calculates the number of users based on occupancy level.
    generate_users(): Generates user attributes such as position and clothing level.
    generate_user_feedback(): Calculates user feedback based on PMV values.
    calculate_pmv(met, clo): Calculates the Predicted Mean Vote (PMV) for a user.
    step(action, adversary_action): Executes a simulation step with given actions.
    reset(): Resets the environment to its initial state.
    render(): Renders the environment (not implemented).
"""
class BuildingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, alpha=1, beta=50, EER=11, seed=42):
        super(BuildingEnv, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.EER = EER
        
        #setting seed
        self.seed_value = seed
        random.seed(seed)
        np.random.seed(seed)
        manual_seed(seed)
        self.rng = np.random.RandomState(seed)  

        self.room_width, self.room_height = 800, 600
        self.action_space = spaces.Discrete(9) 
        low = np.array([-10.0, 0.0, -3.0]) 
        high = np.array([50.0, 1.0, 3.0])
        self.observation_space = spaces.Box(low, high)
        
        self.adversary_action_space = spaces.Discrete(3)

        self.time_step = 0
        self.max_steps = 48  

        self.indoor_temp = 26.0
        self.setpoint_temp = 26.0
        self.outdoor_temp = self.generate_outdoor_temperature()
        self.occupancy_level = self.generate_occupancy_level()

        self.num_users = self.generate_num_users()
        self.users = self.generate_users()

        self.user_feedback = self.generate_user_feedback()
        self.hvac_power = 0.0

        self.history = {
            'indoor_temp': [], 'outdoor_temp': [], 'hvac_power': [],
            'user_feedback': [], 'occupancy': [], 'time': []
        }

    def generate_outdoor_temperature(self):
        time_of_day = (self.time_step % self.max_steps) / self.max_steps * 24.0
        temp = 28 - 6 * np.cos((time_of_day - 14) * np.pi / 12)  # peaks at 2 PM
        return temp + self.rng.normal(0, 1.0)

    def generate_occupancy_level(self):
        time_of_day = (self.time_step % self.max_steps) / self.max_steps * 24.0
        if 9 <= time_of_day < 12 or 13 <= time_of_day < 17:  # Normal work hours
            occupancy = self.rng.uniform(0.5, 0.8)
        elif 12 <= time_of_day < 13:  # Lunch hour (peak)
            occupancy = self.rng.uniform(0.8, 1.0)
        elif 8 <= time_of_day < 9 or 17 <= time_of_day < 18:  # Transition hours
            occupancy = self.rng.uniform(0.2, 0.5)
        else:  # Off hours
            occupancy = self.rng.uniform(0.0, 0.1)
        return occupancy

    def generate_num_users(self):
        max_users = 1000
        return int(self.occupancy_level * max_users)

    def generate_users(self):
        users = []
        for _ in range(self.num_users):
            user = {
                'x': self.rng.randint(50, self.room_width - 50),
                'y': self.rng.randint(50, self.room_height - 50),
                'met': self.rng.uniform(1.0, 1.5),
                'clo': self.rng.uniform(0.5, 1.0)
            }
            users.append(user)
        return users

    def generate_user_feedback(self):
        if self.num_users == 0:
            return 0
        pmv_values = []
        for user in self.users:
            met = user['met']
            clo = user['clo']
            pmv = self.calculate_pmv(met, clo)
            pmv_values.append(pmv)
        aggregated_pmv = np.mean(pmv_values)
        return np.clip(aggregated_pmv, -3.0, 3.0)

    def calculate_pmv(self, met, clo):
        tdb = self.indoor_temp  # Dry bulb temperature
        tr = tdb  # Mean radiant temperature
        rh = 50.0  # Relative humidity
        v = 0.1  # Air velocity

        pmv_result = pmv_ppd(tdb=tdb, tr=tr, vr=v, rh=rh, met=met, clo=clo, standard='ASHRAE')
        return pmv_result['pmv']

    def step(self, action, adversary_action):
        self.setpoint_temp = 22 + action

        temp_difference = self.indoor_temp - self.setpoint_temp
        self.hvac_power = abs(temp_difference) * 10 / self.EER 

        outdoor_influence = (self.outdoor_temp - self.indoor_temp) * 0.1
        hvac_influence = (self.setpoint_temp - self.indoor_temp) * 0.5
        self.indoor_temp += outdoor_influence + hvac_influence

        self.time_step += 1
        done = self.time_step >= self.max_steps
        self.outdoor_temp = self.generate_outdoor_temperature()
        self.occupancy_level = self.generate_occupancy_level()
        self.num_users = self.generate_num_users()
        self.users = self.generate_users()
        self.user_feedback = self.generate_user_feedback()
        
        if adversary_action == 1:
            self.user_feedback += 0.5
        elif adversary_action == 2:
            self.user_feedback -= 0.5

        self.user_feedback = np.clip(self.user_feedback, -3.0, 3.0)
        if self.num_users == 0:
            reward = -self.beta * (self.hvac_power / 1000.0)  
        else:
            comfort_penalty = abs(self.user_feedback) * self.occupancy_level
            energy_penalty = self.hvac_power / 1000.0
            reward = -(self.alpha * comfort_penalty + self.beta * energy_penalty)
        
        reward = reward / 10.0
        adversary_reward = -reward

        self.history['indoor_temp'].append(self.indoor_temp)
        self.history['outdoor_temp'].append(self.outdoor_temp)
        self.history['hvac_power'].append(self.hvac_power)
        self.history['user_feedback'].append(self.user_feedback)
        self.history['occupancy'].append(self.occupancy_level)
        self.history['time'].append(self.time_step)

        state = np.array([self.outdoor_temp, self.occupancy_level, self.user_feedback], dtype=np.float32)
        return state, reward, done, {}, adversary_reward

    def comfort_energy_rewards(self):
        if self.num_users == 0:
            return 0, self.beta * self.hvac_power / 1000.0
        else:
            comfort_penalty = abs(self.user_feedback) * self.occupancy_level
            energy_penalty = self.hvac_power / 1000.0
            return self.alpha * comfort_penalty, self.beta * energy_penalty

    def reset(self):
        # Reset RNG to initial seed + timestep to maintain determinism across episodes
        self.rng = np.random.RandomState(self.seed_value)
        
        self.time_step = 0
        self.indoor_temp = 26.0
        self.setpoint_temp = 26.0
        self.outdoor_temp = self.generate_outdoor_temperature()
        self.occupancy_level = self.generate_occupancy_level()
        self.num_users = self.generate_num_users()
        self.users = self.generate_users()
        self.user_feedback = self.generate_user_feedback()
        self.hvac_power = 0.0

        for key in self.history.keys():
            self.history[key] = []

        state = np.array([self.outdoor_temp, self.occupancy_level, self.user_feedback], dtype=np.float32)
        return state

    def render(self):
        pass

class RegularisedUndistortedRewardBuildingEnv20(BuildingEnv):
    def __init__(self, energy_threshold=1, alpha=1, beta=50, EER=11, seed=42, *args):
        # Initialize the parent class with required arguments
        super(RegularisedUndistortedRewardBuildingEnv20, self).__init__(
            alpha=alpha,
            beta=beta,
            EER=EER,
            seed=seed,
            *args
        )
        self.energy_threshold=energy_threshold
        self.observed_user_feedback = self.user_feedback
        self.action_space = spaces.Discrete(12)

    def step(self, action, adversary_action):
        self.setpoint_temp = 20 + action * 0.5

        temp_difference = self.indoor_temp - self.setpoint_temp
        self.hvac_power = abs(temp_difference) * 10 / self.EER 

        outdoor_influence = (self.outdoor_temp - self.indoor_temp) * 0.1
        hvac_influence = (self.setpoint_temp - self.indoor_temp) * 0.5
        self.indoor_temp += outdoor_influence + hvac_influence

        self.time_step += 1
        done = self.time_step >= self.max_steps
        self.outdoor_temp = self.generate_outdoor_temperature()
        self.occupancy_level = self.generate_occupancy_level()
        self.num_users = self.generate_num_users()
        self.users = self.generate_users()
        self.user_feedback = self.generate_user_feedback()
        
        if adversary_action == 1:
            self.observed_user_feedback = 0.5 + self.user_feedback
        elif adversary_action == 2:
            self.observed_user_feedback = -0.5 + self.user_feedback 

        self.observed_user_feedback = np.clip(self.observed_user_feedback, -3.0, 3.0)
        if self.num_users == 0:
            reward = -self.beta * (max(self.hvac_power - self.energy_threshold, 0) / 1000.0)  
        else:
            comfort_penalty = abs(self.user_feedback) * self.occupancy_level # Rewards are calculated based on true user comfort and not the comfort after distortion by adversary
            energy_penalty = max(self.hvac_power - self.energy_threshold, 0) / 1000.0
            reward = -(self.alpha * comfort_penalty + self.beta * energy_penalty)
        
        reward = reward / 10.0
        adversary_reward = -reward

        self.history['indoor_temp'].append(self.indoor_temp)
        self.history['outdoor_temp'].append(self.outdoor_temp)
        self.history['hvac_power'].append(self.hvac_power)
        self.history['user_feedback'].append(self.user_feedback)
        self.history['occupancy'].append(self.occupancy_level)
        self.history['time'].append(self.time_step)

        state = np.array([self.outdoor_temp, self.occupancy_level, self.observed_user_feedback], dtype=np.float32) # Agent only observed distorted state signals
        return state, reward, done, {}, adversary_reward
    
