"""This module contains the class to store data products in 3D datacube or 2D detector
image."""

__author__ = "ajshajib"


class Data(object):
    """A class to store spectroscopic data."""

    def __init__(self, wavelengths, spectra, spectra_unit, mask=None, noise=None):
        """
        :param wavelengths: wavelengths of the data
        :param spectra: spectra of the data
        :param spectra_unit: unit of the spectra
        :param mask: mask of the data
        :param noise: noise of the data
        """
        self._wavelengths = wavelengths
        self._spectra = spectra
        self._spectra_unit = spectra_unit
        self._mask = mask
        self._noise = noise

    @property
    def spectra(self):
        """Return the spectra of the data."""
        if hasattr(self, '_spectra'):
            return self._spectra
        else:
            return None
        
    @property
    def wavelengths(self):
        """Return the wavelengths of the data."""
        if hasattr(self, '_wavelegnths'):
            return self._wavelegnths
        
    @property
    def spectra_unit(self):
        """Return the unit of the spectra."""
        if hasattr(self, '_spectra_unit'):
            return self._spectra_unit
        else:
            return None
        
    @property
    def mask(self):
        """Return the mask of the data."""
        if hasattr(self, '_mask'):
            return self._mask
        else:
            return None
        
    @property
    def noise(self):
        """Return the noise of the data."""
        if hasattr(self, '_noise'):
            return self._noise
        else:
            return None


class Datacube(Data):
    """A class to store in 3D IFU datacubes."""

    def __init__(self, wavelengths, spectra, spectra_unit, mask=None, noise=None):
        """
        :param wavelengths: wavelengths of the data
        :param spectra: spectra of the data
        :param spectra_unit: unit of the spectra
        :param mask: mask of the data
        :param noise: noise of the data
        """
        super(Datacube, self).__init__(wavelengths, spectra, spectra_unit, mask, noise)