from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from gym_examples.utils.constants import *
from gym_examples.utils.thermal_comfort_model_sim import ThermalComfortModelSim
import random

class PopulationSimulation():
    """
    Human population simulation (collection of humans) and reward calculation of the system
    self.interval: time interval between each step in minutes
    metabolic (0.8-4 limit) and clothing insulation (0-1 limit) based on the ISO standards https://pythermalcomfort.readthedocs.io/en/latest/reference/pythermalcomfort.html.
        met_mu (int): Mean metabolic rate
        met_sigma (int): Standard deviation of metabolic rate
        clo_mu (int): Mean clothing insulation
        clo_sigma (int): Standard deviation of clothing insulation
        move_mu (int): Mean number of people entering/exiting the system at each time step
        move_sigma (int): Standard deviation number of people entering/exiting the system at each time step
    """
    def __init__(self, 
                #  total_human_in_simulation:int=0,
                met_mu: int,
                met_sigma: int,
                clo_mu: int,
                clo_sigma: int, 
                move_mu: int,
                move_sigma:int,
                time_interval:timedelta=timedelta(hours=1)
                ):
        self.interval = time_interval
        self.met_mu = met_mu
        self.met_sigma = met_sigma
        self.clo_mu = clo_mu
        self.clo_sigma = clo_sigma
        self.move_mu = move_mu
        self.move_sigma = move_sigma
        self.comfort_model = ThermalComfortModelSim()

        # Initialise the population of all possible humans in simulation (which will randomly enter and exit the building), first value cannot be negative
        n_people = max(0,(int(random.gauss(self.move_mu, self.met_sigma)))) 
        self.humans = self.generate_humans(n_people)
    
    def reset(self):
        """Reset population"""
        n_people = max(0,(int(random.gauss(self.move_mu, self.met_sigma)))) 
        self.humans = self.generate_humans(n_people)

    def get_df(self) -> pd.DataFrame:
        """ 
        Return the humans dataframe: index, mode, color, human
        Utility function
        """
        return self.humans.copy(deep=True) # Humans cannot be changed
    
    def get_comfort_profile_df(self) -> pd.DataFrame:
        """
        Return the a met clo df padded to padded by MAX_POP_SIZE  x 2
        """
        df = self.humans.loc[:,['met', 'clo']]
        from gym_examples.hitl_mdp_env import HITLAirconEnvironment
        df = df.iloc[:HITLAirconEnvironment.MAX_POP_SIZE] # If there are more people than MAX_POP_SIZE

        pad = max(HITLAirconEnvironment.MAX_POP_SIZE - df.shape[0], 0)
        pad_df = pd.DataFrame(np.empty((pad,2)), columns=["met", "clo"])

        df = pd.concat([df, pad_df], axis=0)
        return df
    
    def step(self) -> None:
        """Update the population simulation by one step"""
        n_exit = min(max(0,(int(random.gauss(self.move_mu, self.move_sigma)))), len(self.humans)) # people entering the building
        exiting_humans = self.humans.sample(n_exit).index
        self.humans = self.humans.drop(exiting_humans)

        n_enter = max(0,(int(random.gauss(self.move_mu, self.move_sigma)))) # people exiting the building
        new_humans = self.generate_humans(n_enter)
        self.humans = pd.concat([self.humans, new_humans])
    
    def get_pmv(self, temp:int) -> np.ndarray[int]:
        """Return the list of PMV counts for -3 to +3 from the population in the building, index 0 is count for -3"""
        pmv = self.humans.apply(lambda row: self.comfort_model.pmv(row['met'], row['clo'], temp), axis=1)
        rounded_pmv = np.round(pmv).astype(int)
        adjusted_indices = rounded_pmv + ThermalComfortModelSim.max_pmv # bincount only works with pos int
        valid_indices = (adjusted_indices >= 0) & (adjusted_indices < ThermalComfortModelSim.max_pmv*2+1)  # NOTE: Removed extreme PMV values which are >3 and < -3
        pmv_counts = np.bincount(adjusted_indices[valid_indices], minlength=ThermalComfortModelSim.max_pmv*2+1)
        return pmv_counts

    def generate_humans(self, n:int) -> pd.DataFrame:
        """Generate a dataframe of size n with human attributes
        ASHRAE 55 2020 limits are 10 < tdb [°C] < 40, 10 < tr [°C] < 40, 0 < vr [m/s] < 2, 1 < met [met] < 4, and 0 < clo [clo] < 1.5
        Reference: https://pythermalcomfort.readthedocs.io/en/latest/reference/pythermalcomfort.html#pythermalcomfort.utilities.running_mean_outdoor_temperature
        """
        met_arr = np.clip(np.random.normal(self.met_mu, self.met_sigma, n), 1, 4)
        clo_arr = np.clip(np.random.normal(self.clo_mu, self.clo_sigma, n), 0, 1.5)
        df = pd.DataFrame({
            'met': met_arr,
            'clo': clo_arr,
        })
        df['color'] = [AGENT_IN_COLOR for _ in df.index]

        def generate_x_y(row):
            """Generating x and y values for visualisation"""
            row['x'] = np.random.rand() * (RECT_WIDTH - TEXT_BOX_LENGTH)
            row['y'] = np.random.rand() * (RECT_LENGTH - TEXT_BOX_HEIGHT) + (SCREENHEIGHT - RECT_LENGTH)
            return row
        df = df.apply(generate_x_y, axis=1)

        return df

