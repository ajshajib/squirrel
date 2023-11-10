"""
This module contains the class to compute the likelihood function.
"""
import jax.numpy as np

class Likelihood(object):
    def __init__(self, data, covariance, model):
        self.data = data
        self.model = model
        self.covariance = covariance
        self.inverse_covariance = np.linalg.inv(self.covariance)

    def compute_log_likelihood(self, params):
        predicted_data = self.model.simulate(params)

        diff = self.data - predicted_data

        log_likelihood = -0.5 * np.dot(diff, np.dot(self.inverse_covariance, diff))

        return log_likelihood

