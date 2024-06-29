from pythermalcomfort.models import pmv_ppd
import numpy as np
import matplotlib.pyplot as plt

class ThermalComfortModelSim():
    def __init__(self):
        self.vr = 0.1 # m/s
        self.rh = 50
        self.met = 1.0
        self.clo = 0.5
        self.function_mapping = None
        self.map_functions()
        self.pmv_scores = None

    def map_functions(self):
        modes = []
        for met in [1.1, 1.2, 1.3]:
            for clo in [0.46, 0.57, 0.68]:
                modes.append({"met": met, "clo": clo})
        self.function_mapping = {i: function for i, function in enumerate(modes)}

    def pmv(self, mode, temp):
        length = len(mode)
        met, clo = [], []
        for m in mode:
            mapping = self.function_mapping[m]
            met.append(mapping['met'])
            clo.append(mapping['clo'])

        self.pmv_scores = pmv_ppd([temp]*length, 
                                    [temp]*length, 
                                    [self.vr]*length, 
                                    [self.rh]*length, 
                                    [met], 
                                    [clo], 
                                    standard="ASHRAE")['pmv'].reshape(-1)
        return self.pmv_scores

    def predict_score(self, mode, temp):
        rst = self.predict_vote(mode, temp)
        if len(rst) == 0:
            return 0
        return sum(rst)

    # def predict_vote(self, mode, temp):
    #     pmvs = self.pmv(mode, temp)
    #     print('here: ', pmvs)
    #     votes = np.where((pmvs > -0.5) & (pmvs < 0.5), pmvs, 0)
    #     votes = np.where(votes < -0.5, votes, -1)
    #     votes = np.where(votes > 0.5, votes, -1)
    #     return votes
    
    def predict_vote(self, mode, temp):
        pmvs = self.pmv(mode, temp)
        print('values: ' , pmvs)
        votes = np.where((pmvs >= -0.5) & (pmvs <= 0.5), 0, -1)
        return votes

    def show_distributions(self):
        x = [temp for temp in np.arange(16, 32, 0.01)]
        plt.figure(figsize=(8, 6))
        legend = []
        for dict_ in self.function_mapping.values():
            met, clo = dict_['met'], dict_['clo']
            y = [pmv_ppd(tdb, tdb, self.vr, self.rh, met, clo, standard="ASHRAE")['pmv'] for tdb in x]
            plt.plot(x, y, "-", lw=1.5)
            legend.append(f"(met, clo)=({met}, {clo})")
        plt.xlabel("Temperature (Celsius)", fontsize=18)
        plt.ylabel("PMV", fontsize=18)
        plt.legend(legend)
        plt.show()