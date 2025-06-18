import pytest
import numpy as np
import numpy.testing as npt
from squirrel.template import Template


class TestTemplate:
    def setup_method(self):
        """Set up the test environment.

        This method is called before every test method to set up any state that is
        shared across tests. This method is called before every test method to set up
        any state that is shared across tests.
        """
        # Initialize wavelengths array
        self.wavelengths = np.array([1, 2, 3])

        # Initialize flux array
        self.flux = np.array([[4, 5, 6], [7, 8, 9]])

        # Define units for flux and wavelength
        self.flux_unit = "arbitrary unit"
        self.wavelength_unit = "nm"

        # Initialize noise array with zeros
        self.noise = np.zeros_like(self.flux)

        # Define full width at half maximum (FWHM)
        self.fwhm = 2.0

        # Create a Template object with the initialized data
        self.template = Template(
            self.wavelengths,
            self.flux,
            self.wavelength_unit,
            self.fwhm,
        )

    def test_flux(self):
        """Test the flux attribute of the Template object.

        This method checks if the flux attribute of the Template object matches the
        expected flux array.
        """
        # Assert that the flux attribute of the template matches the expected flux array
        npt.assert_array_equal(self.template.flux, self.flux)

    def test_wavelengths(self):
        """Test the wavelengths attribute of the Template object.

        This method checks if the wavelengths attribute of the Template object matches
        the expected wavelengths array.
        """
        # Assert that the wavelengths attribute of the template matches the expected wavelengths array
        npt.assert_array_equal(self.template.wavelengths, self.wavelengths)

    def test_merge(self):
        """Test the merge method of the Template class.

        This method creates two Template objects and merges them using the merge method.
        It then checks if the merged Template object has the expected attributes.
        """
        # Create mock data for the test
        wavelengths = np.linspace(4000, 5000, 1000)
        flux1 = np.random.normal(1, 0.1, (1000, 5))
        flux2 = np.random.normal(1, 0.1, (1000, 3))
        wavelength_unit = "AA"
        fwhm = 2.0

        # Create two Template objects with the mock data
        template1 = Template(wavelengths, flux1, wavelength_unit, fwhm)
        template2 = Template(wavelengths, flux2, wavelength_unit, fwhm)

        # Merge the two Template objects
        merged_template = template1.merge(template2)

        # Assertions to check the output
        assert isinstance(merged_template, Template)
        assert merged_template.flux.shape == (1000, 8)
        np.testing.assert_equal(merged_template.wavelengths, wavelengths)

        # Test with single row template
        flux3 = np.random.normal(1, 0.1, (1, 1000))
        template3 = Template(wavelengths, flux3, wavelength_unit, fwhm)
        merged_template_single = template2.merge(template3)
        assert merged_template_single.flux.shape == (1000, 4)

        merged_template_single = template3.merge(template2)
        assert merged_template_single.flux.shape == (1000, 5)

    def test_and_operator(self):
        """Test the __and__ method (overloaded & operator) of the Template class.

        This method creates two Template objects and merges them using the & operator.
        It then checks if the merged Template object has the expected attributes.
        """
        # Create mock data for the test
        wavelengths = np.linspace(4000, 5000, 1000)
        flux1 = np.random.normal(1, 0.1, (1000, 5))
        flux2 = np.random.normal(1, 0.1, (1000, 3))
        wavelength_unit = "AA"
        fwhm = 2.0

        # Create two Template objects with the mock data
        template1 = Template(wavelengths, flux1, wavelength_unit, fwhm)
        template2 = Template(wavelengths, flux2, wavelength_unit, fwhm)

        # Merge the two Template objects using the & operator
        merged_template = template1 & template2

        # Assertions to check the output
        assert isinstance(merged_template, Template)
        assert merged_template.flux.shape == (1000, 8)
        np.testing.assert_equal(merged_template.wavelengths, wavelengths)

    def test_iand_operator(self):
        """Test the __iand__ method (overloaded &= operator) of the Template class.

        This method creates two Template objects and merges them using the &= operator.
        It then checks if the merged Template object has the expected attributes.
        """
        # Create mock data for the test
        wavelengths = np.linspace(4000, 5000, 1000)
        flux1 = np.random.normal(1, 0.1, (1000, 5))
        flux2 = np.random.normal(1, 0.1, (1000, 3))
        wavelength_unit = "AA"
        fwhm = 2.0

        # Create two Template objects with the mock data
        template1 = Template(wavelengths, flux1, wavelength_unit, fwhm)
        template2 = Template(wavelengths, flux2, wavelength_unit, fwhm)

        # Merge the two Template objects using the &= operator
        template1 &= template2

        # Assertions to check the output
        assert isinstance(template1, Template)
        assert template1.flux.shape == (1000, 8)
        np.testing.assert_equal(template1.wavelengths, wavelengths)

    def test_combine_weighted(self):
        """Test the combine_weighted method of the Template class.

        This method creates a Template object and combines its flux values using the
        combine_weighted method with specified weights. It then checks if the combined
        Template object has the expected attributes.
        """
        # Create mock data for the test
        wavelengths = np.linspace(4000, 5000, 1000)
        flux = np.random.normal(1, 0.1, (1000, 5))
        weights = np.random.rand(5)
        wavelength_unit = "AA"
        fwhm = 2.0

        # Create a Template object with the mock data
        template = Template(wavelengths, flux, wavelength_unit, fwhm)

        # Combine the flux values using the specified weights
        combined_template = template.combine_weighted(weights)

        # Assertions to check the output
        assert isinstance(combined_template, Template)
        assert combined_template.flux.shape == (1000, 1)
        np.testing.assert_equal(combined_template.wavelengths, wavelengths)


if __name__ == "__main__":
    pytest.main()
