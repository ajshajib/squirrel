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

    def test_merge(self):
        # Create mock data for the test
        wavelengths = np.linspace(4000, 5000, 1000)
        flux1 = np.random.normal(1, 0.1, (1000, 5))
        flux2 = np.random.normal(1, 0.1, (1000, 3))
        wavelength_unit = "AA"
        fwhm = 2.0

        # Create two Template objects
        template1 = Template(wavelengths, flux1, wavelength_unit, fwhm)
        template2 = Template(wavelengths, flux2, wavelength_unit, fwhm)

        # Call the merge method
        merged_template = template1.merge(template2)

        # Assertions to check the output
        assert isinstance(merged_template, Template)
        assert merged_template.flux.shape == (1000, 8)
        np.testing.assert_equal(merged_template.wavelengths, wavelengths)

    def test_and_operator(self):
        # Create mock data for the test
        wavelengths = np.linspace(4000, 5000, 1000)
        flux1 = np.random.normal(1, 0.1, (1000, 5))
        flux2 = np.random.normal(1, 0.1, (1000, 3))
        wavelength_unit = "AA"
        fwhm = 2.0

        # Create two Template objects
        template1 = Template(wavelengths, flux1, wavelength_unit, fwhm)
        template2 = Template(wavelengths, flux2, wavelength_unit, fwhm)

        # Call the __and__ method
        merged_template = template1 & template2

        # Assertions to check the output
        assert isinstance(merged_template, Template)
        assert merged_template.flux.shape == (1000, 8)
        np.testing.assert_equal(merged_template.wavelengths, wavelengths)

    def test_iand_operator(self):
        # Create mock data for the test
        wavelengths = np.linspace(4000, 5000, 1000)
        flux1 = np.random.normal(1, 0.1, (1000, 5))
        flux2 = np.random.normal(1, 0.1, (1000, 3))
        wavelength_unit = "AA"
        fwhm = 2.0

        # Create two Template objects
        template1 = Template(wavelengths, flux1, wavelength_unit, fwhm)
        template2 = Template(wavelengths, flux2, wavelength_unit, fwhm)

        # Call the __iand__ method
        template1 &= template2

        # Assertions to check the output
        assert isinstance(template1, Template)
        assert template1.flux.shape == (1000, 8)
        np.testing.assert_equal(template1.wavelengths, wavelengths)

    def test_combine_weighted(self):
        # Create mock data for the test
        wavelengths = np.linspace(4000, 5000, 1000)
        flux = np.random.normal(1, 0.1, (1000, 5))
        weights = np.random.rand(5)
        wavelength_unit = "AA"
        fwhm = 2.0

        # Create a Template object
        template = Template(wavelengths, flux, wavelength_unit, fwhm)

        # Call the combine_weighted method
        combined_template = template.combine_weighted(weights)

        # Assertions to check the output
        assert isinstance(combined_template, Template)
        assert combined_template.flux.shape == (1000, 1)
        np.testing.assert_equal(combined_template.wavelengths, wavelengths)


if __name__ == "__main__":
    pytest.main()
