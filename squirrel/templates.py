"""
This module contains the class to store stellar and other templates and process them.
"""
import jax.numpy as np

class Template(object):
    def __init__(self, values, wavelengths):
        self._values = values
        self._wavelengths = wavelengths

        if wavelengths[1] / wavelengths[0] == wavelengths[2] / wavelengths[1]:
            self._scaling = 'log'
        else:
            self._scaling = 'linear'
    @property
    def value(self):
        return self._values

    @property
    def wavelength(self):
        return self._wavelengths

    @property
    def scaling(self):
        """log or linearly sampled wavelength grid"""
        return self._scaling

    def interp(self, l):
        return np.interp(l, self._wavelengths, self._values)