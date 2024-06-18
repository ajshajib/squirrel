"""This module contains the class to store stellar and other templates and process
them."""

import numpy as np
from copy import deepcopy
from .data import Spectra


class Template(Spectra):
    """A class to store stellar and other templates and process them."""

    def __init__(
        self,
        wavelengths,
        flux,
        wavelength_unit,
        fwhm,
        flux_unit="arbitrary",
    ):
        """Initialize the template.

        :param wavelengths: wavelengths of the template
        :type wavelengths: numpy.ndarray
        :param flux: flux of the template
        :type flux: numpy.ndarray
        :param wavelength_unit: unit of the wavelength
        :type wavelength_unit: str
        :param fwhm: full width at half maximum
        :type fwhm: float
        :param flux_unit: unit of the flux
        :type flux_unit: str
        :param noise: noise of the template
        :type noise: numpy.ndarray
        """
        if len(flux.shape) == 1:
            flux = flux[:, np.newaxis]

        super(Template, self).__init__(
            wavelengths=wavelengths,
            flux=flux,
            wavelength_unit=wavelength_unit,
            fwhm=fwhm,
            z_lens=0.0,
            z_source=0.0,
            flux_unit=flux_unit,
            noise=None,
        )

    def merge(self, other):
        """Merge the template with another template.

        :param template: template to merge with
        :type template: squirrel.template.Template
        """
        assert (
            self.wavelength_unit == other.wavelength_unit
        ), "Wavelength units do not match"
        assert self.fwhm == other.fwhm, "FWHM do not match"
        np.testing.assert_equal(
            self.wavelengths, other.wavelengths, err_msg="Wavelengths do not match"
        )

        new_template = deepcopy(self)
        new_template.flux = np.concatenate((self.flux, other.flux))
        self.noise = np.concatenate((self.noise, other.noise))

        return new_template

    def __and__(self, other):
        """Merge the template with another template.

        :param template: template to merge with
        :type template: squirrel.template.Template
        """
        new_template = deepcopy(self)
        return new_template.merge(other)

    def __iand__(self, other):
        """Merge the template with another template.

        :param template: template to merge with
        :type template: squirrel.template.Template
        """
        return self.merge(other)

    def combine_weighted(self, weights):
        """Combine the templates into one single template.

        :param weights: weights for each template
        :type weights: numpy.array
        """
        new_template = deepcopy(self)
        flux = new_template.flux @ weights
        flux /= np.median(flux)

        if len(flux.shape) == 1:
            flux = flux[:, np.newaxis]

        new_template.flux = flux

        return new_template
