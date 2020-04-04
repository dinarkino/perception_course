"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle


def LWS(weights, num_particles):
    cum = list(np.cumsum(np.array(weights) / np.sum(weights)))
    length = len(cum)
    indx = []
    U = (np.random.rand() + np.arange(num_particles)) / num_particles
    i = 0
    for u in U:
        while u > cum[i] and i < length:
            i += 1
        indx += [i - 1]
    return indx

class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, beta)
        
        self.particles = np.ones((num_particles, 3)) * initial_state.mu.reshape(1, -1).ravel()
        self.num_particles = num_particles

    def predict(self, u):
        for particle in range(self.num_particles):
            self.particles[particle] = sample_from_odometry(self.particles[particle], u, self._alphas)
            
        self._state_bar.mu = get_gaussian_statistics(self.particles).mu
        self._state_bar.Sigma = get_gaussian_statistics(self.particles).Sigma

    def update(self, z):
        bearings = np.zeros(self.num_particles)
        for particle in range(self.num_particles):
            bearings[particle], landmark_id = get_observation(self.particles[particle], z[1])
            bearings[particle] = wrap_angle(bearings[particle] - z[0])
        weights = gaussian().pdf(bearings / np.sqrt(self._Q))
        
        self.particles = self.particles[LWS(weights, self.num_particles)]
        self._state.mu = get_gaussian_statistics(self.particles).mu
        self._state.Sigma = get_gaussian_statistics(self.particles).Sigma
    
    # for show_particles
    @property
    def X(self):
        return self.particles
