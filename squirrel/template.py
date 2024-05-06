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
        noise=None,
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
        super(Template, self).__init__(
            wavelengths=wavelengths,
            flux=flux,
            wavelength_unit=wavelength_unit,
            fwhm=fwhm,
            z_lens=0.0,
            z_source=0.0,
            flux_unit=flux_unit,
            noise=noise if noise else np.zeros_like(flux),
        )

    def merge(self, template):
        """Merge the template with another template.

        :param template: template to merge with
        :type template: squirrel.template.Template
        """
        assert self.wavelength_unit == template.wavelength_unit
        assert self.fwhm == template.fwhm
        np.testing.assert_equal(self.wavelengths, template.wavelengths)

        new_template = deepcopy(self)
        new_template.flux = np.concatenate((self.flux, template.flux))
        self.noise = np.concatenate((self.noise, template.noise))

        return new_template
