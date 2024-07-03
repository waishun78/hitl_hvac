import numpy as np
from datetime import timedelta
from constants import *

class PopulationSim():
    def __init__(self, sample_size=500, time_interval=timedelta(hours=1), is_debug=False):
        self.normal_distribution = []
        self.pop_distribution = []
        self.sample_size = sample_size
        self.time_interval = time_interval
        self.curr_time = timedelta(days=-1, hours=DAY_END_TIME)
        self.is_debug = is_debug

    def generateDistribution(self):
        mu = (DAY_START_TIME + DAY_END_TIME) / 2
        sigma = (DAY_END_TIME - DAY_START_TIME) / 6
        self.normal_distribution = np.random.normal(mu, sigma, self.sample_size)
        interval = self.time_interval / timedelta(hours=1)
        l = np.arange(DAY_START_TIME, DAY_END_TIME, interval)
        for i in l:
            self.pop_distribution.append(len(self.normal_distribution[(self.normal_distribution > i) & (self.normal_distribution < i + interval)]))

    def update(self, actual_in_num):
        if len(self.pop_distribution) == 0:
            self.generateDistribution()
            self.curr_time += timedelta(hours=14)
        
        if self.is_debug:
            print("pop_simulation:", self.pop_distribution)

        expect_in_num = self.pop_distribution.pop(0)
        delta_in_num = abs(expect_in_num - actual_in_num)
        epislon = np.random.randint(0, actual_in_num // 8) if actual_in_num // 8 > 0 else 0 
        if expect_in_num > actual_in_num:
            move_in_num = delta_in_num + epislon
            move_out_num = (actual_in_num + move_in_num) - expect_in_num
        else:
            move_out_num = delta_in_num + epislon
            move_in_num = expect_in_num - (actual_in_num - move_out_num)
        self.curr_time += self.time_interval

        if self.is_debug:
            print(f"{self.curr_time - self.time_interval} ~ {self.curr_time}",
                "\n expected_in:", expect_in_num,
                "\n delta_in:", delta_in_num)
        
        return move_in_num, move_out_num
