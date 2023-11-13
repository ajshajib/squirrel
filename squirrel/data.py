"""This module contains the class to store data products in 3D datacube or 2D detector
image."""


class Data(object):
    def __init__(self):
        pass


class Datacube(Data):
    def __init__(self):
        pass


class Data2D(Data):
    def __init__(self, image, mask, noise, wavelength, x, y):
        self._image = image
        self._mask = mask
        self._noise = noise
        self._wavelength = wavelength
        self._x = x
        self._y = y

    @property
    def flux(self):
        return self._image

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
