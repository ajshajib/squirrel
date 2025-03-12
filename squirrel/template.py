"""This module contains the class to store stellar and other templates and
process them."""

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
        """
        # Ensure the flux array has at least two dimensions
        if len(flux.shape) == 1:
            flux = flux[:, np.newaxis]

        # Initialize the parent class Spectra with the provided parameters
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

        :param other: template to merge with
        :type other: squirrel.template.Template
        :return: A new Template instance with merged flux
        :rtype: squirrel.template.Template
        """
        # Ensure the wavelength units and FWHM match
        assert (
            self.wavelength_unit == other.wavelength_unit
        ), "Wavelength units do not match"
        assert self.fwhm == other.fwhm, "FWHM do not match"

        # Ensure the wavelengths match
        np.testing.assert_equal(
            self.wavelengths, other.wavelengths, err_msg="Wavelengths do not match"
        )

        # Create a deep copy of the current template
        new_template = deepcopy(self)

        # Concatenate the flux arrays along the second axis
        new_template.flux = np.concatenate((self.flux, other.flux), axis=1)

        return new_template

    def __and__(self, other):
        """Merge the template with another template using the & operator.

        :param other: template to merge with
        :type other: squirrel.template.Template
        :return: A new Template instance with merged flux
        :rtype: squirrel.template.Template
        """
        # Create a deep copy of the current template and merge with the other template
        new_template = deepcopy(self)
        return new_template.merge(other)

    def __iand__(self, other):
        """Merge the template with another template using the &= operator.

        :param other: template to merge with
        :type other: squirrel.template.Template
        :return: The current Template instance with merged flux
        :rtype: squirrel.template.Template
        """
        # Merge the current template with the other template
        return self.merge(other)

    def combine_weighted(self, weights):
        """Combine the templates into one single template using weighted sum.

        :param weights: weights for each template
        :type weights: numpy.array
        :return: A new Template instance with combined flux
        :rtype: squirrel.template.Template
        """
        # Create a deep copy of the current template
        new_template = deepcopy(self)

        # Compute the weighted sum of the flux arrays
        flux = new_template.flux @ weights

        # Normalize the flux by its median value
        flux /= np.median(flux)

        # Ensure the flux array has at least two dimensions
        if len(flux.shape) == 1:
            flux = flux[:, np.newaxis]

        # Update the flux of the new template
        new_template.flux = flux

        return new_template
