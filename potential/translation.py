import math
import numpy as np
from datetime import timedelta
import gym
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Opion 1
# linear translation class
# The idea being since the transformation is linear it keeps the relative differences between values within the same variables?
# Hvae not integrated with the RL just yet.
class LinearTranslation(gym.Env):
    def __init__(self, original_env, new_env):
        self.original_env= original_env
        self.new_env= new_env

    def translate(self, obs):
        translated_obs= obs.copy()
        translated_obs['met'] = self.translate_met(obs['met'])
        translated_obs['clo'] = self.translate_clo(obs['clo'])
        return translated_obs
    
    def translate_met(self, met):
        return (met - self.new_env['met_mu']) / self.new_env['met_sigma'] * \
               self.original_env['met_sigma'] + self.original_env['met_mu']

    def translate_clo(self, clo):
        return (clo - self.new_env['clo_mu']) / self.new_env['clo_sigma'] * \
               self.original_env['clo_sigma'] + self.original_env['clo_mu']

# Option 2
# kde transform
# kde-based translation can handle non-linear relationships between the original and new distributions. This makes sense iaf the new environment data does not follow a normal distribution or if the distributions have differentshapes
class KDETranslation(gym.Env):
    def __init__(self, original_env , new_env):
        self.original_env = original_env
        self.new_env= new_env
        self.kde_met = None
        self.kde_new_met = None
        self.kde_clo = None
        self.kde_new_clo = None
        
    def fit(self, original_data, new_data):
        self.kde_original_met = gaussian_kde(original_data['met'])
        self.kde_new_met = gaussian_kde(new_data['met'])
        self.kde_original_clo = gaussian_kde(original_data['clo'])
        self.kde_new_clo = gaussian_kde(new_data['clo'])

    def translate(self, obs):
        translated_obs= obs.copy()
        translated_obs['met'] = self.translate_met(obs['met'])
        translated_obs['clo'] = self.translate_clo(obs['clo'])
        return translated_obs
    
    def translate_met(self, met):
        return self.kde_transform(self.kde_original_met, self.kde_new_met, met)

    def translate_clo(self, clo):
        return self.kde_transform(self.kde_original_clo, self.kde_new_clo, clo)

    def kde_transform(self, kde_original, kde_new, x):
        cdf_value = kde_new.integrate_box_1d(-np.inf, x)
        return kde_original.ppf(cdf_value)
