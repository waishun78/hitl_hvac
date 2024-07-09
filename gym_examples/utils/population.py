from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from gym_examples.utils.constants import *
from gym_examples.utils.thermal_comfort_model_sim import ThermalComfortModelSim
import random

class Human(object):
    """
    Human - randomly generate position and vote function
    """
    def __init__(self, id, mode, x=0, y=0):
        self.id = id
        self.mode = mode
        self.x = x
        self.y = y
        self.thermal_model = ThermalComfortModelSim()
    
    def random_in(self):
        self.color = AGENT_IN_COLOR
        self.x = np.random.rand() * (RECT_WIDTH - TEXT_BOX_LENGTH)
        self.y = np.random.rand() * (RECT_LENGTH - TEXT_BOX_HEIGHT) + (SCREENHEIGHT - RECT_LENGTH)
        return self

    def random_out(self):
        self.color = AGENT_OUT_COLOR
        self.x = np.random.rand() * (RECT_WIDTH - TEXT_BOX_LENGTH) + (SCREENWIDTH - RECT_WIDTH)
        self.y = np.random.rand() * (RECT_LENGTH - TEXT_BOX_HEIGHT) + (SCREENHEIGHT - RECT_LENGTH)
        return self

    def update_position(self, new_x, new_y):
        self.x = new_x
        self.y = new_y
        return self
    
    def get_vote(self, temp) -> int:
        """Given a temperature, find the vote of the human"""
        return self.thermal_model.get_vote(self.mode, temp)

class PopulationSimulation():
    """
    Human population simulation (collection of humans) and reward calculation of the system
    self.interval: time interval between each step in minutes
    """
    def __init__(self, total_human_in_simulation:int=0, time_interval:timedelta=timedelta(hours=1)):
        self.interval = time_interval

        #TODO: Update population simulation to be called by a method to just call the agent instead of initialising it with humans ->

        # Initialise the population of all possible humans in simulation (which will randomly enter and exit the building)
        sample_df = self.sample_population_from_csv(total_human_in_simulation) #TODO: Can replace this with some other distribution generation method
        self.group_data_df = sample_df.copy()
        self.group_data_df.reset_index(inplace=True, drop=True)
        self.group_data_df.rename(columns={'user':'mode'}, inplace=True)

        self.group_data_df['color'] = [AGENT_IN_COLOR for _ in self.group_data_df.index]
        self.group_data_df['human'] = self.group_data_df.apply(lambda row: Human(id=row.name, mode=row['mode']), axis=1)
        self.group_data_df['inside_building'] = False

        # Randomly initialise the humans inside the building and their positions
        initial_n_humans_inside = random.randint(0,len(self.group_data_df))
        humans_in_building = self.group_data_df.sample(initial_n_humans_inside).index
        self.group_data_df.loc[humans_in_building, 'inside_building'] = True
        self.group_data_df.loc[humans_in_building, 'human'].apply(lambda human: human.random_in())

        # Removed font and screen such that we render it using the render engine
        # agent id is encapsulated in the df id

    def get_df(self) -> pd.DataFrame:
        """ 
        Return the humans dataframe: index, mode, color, human
        Utility function
        """
        return self.group_data_df.copy(deep=True) # Humans cannot be changed

    def sample_population_from_csv(self, n_sample:int) -> pd.DataFrame:
        """Sample human modes from csv given pop_size"""
        data_df = pd.read_csv(DATA_FILE, low_memory=False)
        if n_sample > len(data_df):
            raise ValueError("Group size is larger than the size of the data")
        return data_df.sample(n=n_sample)
    
    def step(self):
        """Update the population simulation by one step"""
        in_building = self.group_data_df[self.group_data_df['inside_building']]
        out_building = self.group_data_df[~self.group_data_df['inside_building']]

        n_exiting = random.randint(0, len(in_building))
        n_entering = random.randint(0, len(out_building))

        # Randomly sample humans to exit and enter the building
        exiting_humans = in_building.sample(n_exiting).index
        entering_humans = out_building.sample(n_entering).index

        # Update the inside_building status
        self.group_data_df.loc[exiting_humans, 'inside_building'] = False
        self.group_data_df.loc[entering_humans, 'inside_building'] = True

        # Call random_in for entering humans and random_out for exiting humans
        self.group_data_df.loc[exiting_humans, 'human'].apply(lambda human: human.random_out())
        self.group_data_df.loc[entering_humans, 'human'].apply(lambda human: human.random_in())

        # self.group_data_df.loc[self.group_data_df['inside_building'], 'human'].apply(lambda human: human.update_position())

    def get_humans(self, in_building:bool) -> List[Human]:
        """Return a list of humans inside/not inside the building"""
        humans = self.group_data_df[self.group_data_df['inside_building']] if in_building else self.group_data_df[~self.group_data_df['inside_building']]
        return list(humans['human'])

    def get_vote_up_count(self, temp: float) -> Tuple[int, int]:
        """Return the vote of a human given the temp temperature"""
        vote_up_count = 0
        vote_down_count = 0
        
        for human in self.group_data_df[self.group_data_df['inside_building']==True]['human']:
            vote = human.get_vote(temp)
            if vote==1: vote_up_count += 1
            elif vote==-1: vote_down_count += 1
        return vote_up_count, vote_down_count

