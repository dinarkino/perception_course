"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.objects import Gaussian
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle
from field_map import FieldMap

def G_jacob(state, motion):
    x, y, theta = state
    drot1, dtrans, drot2 = motion
    G = np.array([[1, 0, -dtrans * np.sin(theta + drot1)], 
                  [0, 1, dtrans * np.cos(theta + drot1)], 
                  [0, 0, 1]])
    return(G)

def V_jacob(state, motion):
    x, y, theta = state
    drot1, dtrans, drot2 = motion
    V = np.array([[-dtrans * np.sin(theta + drot1), np.cos(theta + drot1), 0],
                  [dtrans * np.cos(theta + drot1),  np.sin(theta + drot1), 0], 
                  [1, 0, 1]])
    return(V)

def H_jacob(state, lm_id):
    lm_id = int(lm_id)
    fm = FieldMap()
    x, y, theta = state
    mx, my = fm.landmarks_poses_x[lm_id], fm.landmarks_poses_y[lm_id]
    q = (mx - x)**2 + (my - y)**2
    H = np.array([[(my - y) / q, -(mx - x) / q, -1]])
    return H

class EKF(LocalizationFilter):
    def predict(self, u):
        G = G_jacob(self.mu, u)
        V = V_jacob(self.mu, u)
        mu_ = get_prediction(self.mu, u)
        M = get_motion_noise_covariance(u, self._alphas)
        Sigma_ = G @ self.Sigma @ G.T + V @ M @ V.T
        
        sts = Gaussian(mu_, Sigma_)
        self._state_bar.mu = sts.mu
        self._state_bar.Sigma = sts.Sigma

    def update(self, z):
        bearing, landmark_id = get_expected_observation(self.mu_bar, z[1])
        H = H_jacob(self.mu_bar, z[1]) 
        S = H @ self.Sigma_bar @ H.T + np.array([[self._Q]])
        K = self.Sigma_bar @ H.T @ np.linalg.inv(S)
        mu = (self._state_bar.mu + K @ np.array([[z[0] - bearing]]))[:, 0]
        Sigma = (np.eye(3) - K @ H) @ self.Sigma_bar
        
        sts = Gaussian(mu, Sigma)
        self._state.mu = sts.mu
        self._state.Sigma = sts.Sigma
        
        
