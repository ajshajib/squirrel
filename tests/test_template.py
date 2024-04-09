import pytest
import numpy as np
import numpy.testing as npt
from squirrel.template import Template


class TestTemplate:
    def setup_method(self):
        """Set up the test."""
        self.wavelengths = np.array([1, 2, 3])
        self.flux = np.array([[4, 5, 6], [7, 8, 9]])
        self.flux_unit = "arbitrary unit"
        self.wavelength_unit = "nm"
        self.noise = np.zeros_like(self.flux)
        self.fwhm = 2.0
        self.template = Template(
            self.wavelengths,
            self.flux,
            self.wavelength_unit,
            self.fwhm,
        )

    def test_flux(self):
        npt.assert_array_equal(self.template.flux, self.flux)

    def test_wavelengths(self):
        npt.assert_array_equal(self.template.wavelengths, self.wavelengths)


if __name__ == "__main__":
    pytest.main()
