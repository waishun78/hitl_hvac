from typing import Tuple
from pythermalcomfort.models import pmv_ppd
import numpy as np
import matplotlib.pyplot as plt

class ThermalComfortModelSim():
    def __init__(self):
        self.vr = 0.1 # m/s
        self.rh = 50
        self.met_l = [1.1, 1.2, 1.3]
        self.clo_l = [0.46, 0.57, 0.68]

    def pmv(self, mode:int, temp:float):
        met = self.met_l[mode%3]
        clo = self.clo_l[mode//3]
        pmv = pmv_ppd(temp, temp, self.vr, self.rh, met, clo, standard="ASHRAE")['pmv']
        return pmv
    
    def get_vote(self, mode, temp) -> int:
        """Return vote up (+1) vote down(-1) based on mode and temp"""
        pmv = self.pmv(mode, temp)
        vote = -1 if pmv>0.5 else (1 if pmv<-0.5 else 0)
        return vote

    # def show_distributions(self):
    #     x = [temp for temp in np.arange(16, 32, 0.01)]
    #     plt.figure(figsize=(8, 6))
    #     legend = []
    #     for dict_ in self.function_mapping.values():
    #         met, clo = dict_['met'], dict_['clo']
    #         y = [pmv_ppd(tdb, tdb, self.vr, self.rh, met, clo, standard="ASHRAE")['pmv'] for tdb in x]
    #         plt.plot(x, y, "-", lw=1.5)
    #         legend.append(f"(met, clo)=({met}, {clo})")
    #     plt.xlabel("Temperature (Celsius)", fontsize=18)
    #     plt.ylabel("PMV", fontsize=18)
    #     plt.legend(legend)
    #     plt.show()