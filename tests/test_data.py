import pytest
import numpy as np
import numpy.testing as npt
from copy import deepcopy
from squirrel.data import Spectra
from squirrel.data import Datacube
from squirrel.data import VoronoiBinnedSpectra
from squirrel.data import RadiallyBinnedSpectra


class TestSpectra:
    """Test suite for the Spectra class."""

    def setup_method(self):
        """Setup method to initialize the Spectra object with test data."""
        self.wavelengths = np.array([1, 2, 3])
        self.flux = np.array([4, 5, 6])
        self.flux_unit = "arbitrary unit"
        self.wavelength_unit = "nm"
        self.noise = np.array([0.1, 0.2, 0.3])
        self.covariance = np.diag(self.noise**2)
        self.fwhm = 2.0
        self.z_lens = 0.5
        self.z_source = 1.0
        self.spectra = Spectra(
            self.wavelengths,
            self.flux,
            self.wavelength_unit,
            self.fwhm,
            self.z_lens,
            self.z_source,
            self.flux_unit,
            self.noise,
            self.covariance,
        )

    def test_flux(self):
        """Test the flux property of the Spectra object."""
        npt.assert_array_equal(self.spectra.flux, self.flux)

    def test_flux_setter(self):
        """Test the flux setter method of the Spectra object."""
        self.spectra.flux = np.array([7, 8, 9])
        npt.assert_array_equal(self.spectra.flux, np.array([7, 8, 9]))

    def test_wavelengths(self):
        """Test the wavelengths property of the Spectra object."""
        npt.assert_array_equal(self.spectra.wavelengths, self.wavelengths)

    def test_wavelengths_setter(self):
        """Test the wavelengths setter method of the Spectra object."""
        self.spectra.wavelengths = np.array([4, 5, 6])
        npt.assert_array_equal(self.spectra.wavelengths, np.array([4, 5, 6]))

    def test_wavelength_unit(self):
        """Test the wavelength_unit property of the Spectra object."""
        assert self.spectra.wavelength_unit == self.wavelength_unit

    def test_flux_unit(self):
        """Test the flux_unit property of the Spectra object."""
        assert self.spectra.flux_unit == self.flux_unit

    def test_fwhm(self):
        """Test the fwhm property of the Spectra object."""
        assert self.spectra.fwhm == self.fwhm

    def test_noise(self):
        """Test the noise property of the Spectra object."""
        npt.assert_array_equal(self.spectra.noise, self.noise)

    def test_noise_setter(self):
        """Test the noise setter method of the Spectra object."""
        self.spectra.noise = np.array([0.4, 0.5, 0.6])
        npt.assert_array_equal(self.spectra.noise, np.array([0.4, 0.5, 0.6]))

    def test_covariance(self):
        """Test the covariance property of the Spectra object."""
        npt.assert_array_equal(self.spectra.covariance, self.covariance)

    def test_covariance_setter(self):
        """Test the covariance setter method of the Spectra object."""
        self.spectra.covariance = np.diag([0.4, 0.5, 0.6])
        npt.assert_array_equal(self.spectra.covariance, np.diag([0.4, 0.5, 0.6]))

    def test_z_lens(self):
        """Test the z_lens property of the Spectra object."""
        assert self.spectra.z_lens == self.z_lens

    def test_z_source(self):
        """Test the z_source property of the Spectra object."""
        assert self.spectra.z_source == self.z_source

    def test_spectra_modifications(self):
        """Test the spectra_modifications property of the Spectra object."""
        assert self.spectra.spectra_modifications == []

    def test_spectra_modifications_setter(self):
        """Test the spectra_modifications setter method of the Spectra object."""
        self.spectra.spectra_modifications = "rebinned"
        assert self.spectra.spectra_modifications == "rebinned"

    def test_velocity_scale(self):
        """Test the velocity_scale property of the Spectra object."""
        assert self.spectra.velocity_scale is None

    def test_velocity_scale_setter(self):
        """Test the velocity_scale setter method of the Spectra object."""
        self.spectra.velocity_scale = 1.0
        assert self.spectra.velocity_scale == 1.0

    def test_wavelengths_frame(self):
        """Test the wavelengths_frame property of the Spectra object."""
        assert self.spectra.wavelengths_frame == "observed"

    def test_wavelengths_frame_setter(self):
        """Test the wavelengths_frame setter method of the Spectra object."""
        self.spectra.wavelengths_frame = "rest"
        assert self.spectra.wavelengths_frame == "rest"

    def test_deredshift(self):
        """Test the deredshift method of the Spectra object."""
        self.spectra.deredshift(redshift=1.0)
        npt.assert_equal(self.spectra.wavelengths, np.array([0.5, 1.0, 1.5]))
        assert self.spectra.fwhm == self.fwhm / 2.0

        self.spectra.reset()
        self.spectra.deredshift(target_frame="source")
        npt.assert_equal(self.spectra.wavelengths, np.array([0.5, 1.0, 1.5]))
        assert self.spectra.fwhm == self.fwhm / (1.0 + self.z_source)

        self.spectra.reset()
        self.spectra.deredshift(target_frame="lens")
        npt.assert_equal(
            self.spectra.wavelengths, np.array([1, 2, 3]) / (1 + self.z_lens)
        )
        assert self.spectra.fwhm == self.fwhm / (1.0 + self.z_lens)

        with pytest.raises(ValueError):
            self.spectra.deredshift(target_frame="unknown")

    def test_clip(self):
        """Test the clip method of the Spectra object."""
        self.spectra.clip(wavelength_min=1.5, wavelength_max=2.5)
        npt.assert_equal(self.spectra.wavelengths, np.array([2]))
        npt.assert_equal(self.spectra.flux, np.array([5]))
        npt.assert_equal(self.spectra.noise, np.array([0.2]))

    def test_reset(self):
        """Test the reset method of the Spectra object."""
        self.spectra.deredshift(redshift=1.0)
        self.spectra.wavelengths = None
        self.spectra.flux = None
        self.spectra.spectra_modifications = None
        self.spectra.wavelengths_state = None
        self.spectra.wavelengths_frame = None

        self.spectra.reset()
        npt.assert_equal(self.spectra.wavelengths, [1, 2, 3])
        npt.assert_equal(self.spectra.flux, [4, 5, 6])
        assert self.spectra.fwhm == self.fwhm
        assert self.spectra.spectra_modifications == []
        assert self.spectra.wavelengths_frame == "observed"

    def test_add_function(self):
        """Test the _add method and addition operator of the Spectra object."""
        spectra = Spectra(
            np.array([1, 2, 3]),
            np.array([2, 3, 4]),
            "nm",
            2.0,
            0.5,
            1.0,
            noise=np.array([0.1, 0.2, 0.3]),
        )
        sum_spectra = deepcopy(self.spectra)
        self.spectra._add(spectra, sum_spectra)
        npt.assert_equal(sum_spectra.wavelengths, np.array([1, 2, 3]))
        npt.assert_equal(sum_spectra.flux, np.array([6, 8, 10]))
        npt.assert_equal(sum_spectra.noise, np.sqrt(2) * np.array([0.1, 0.2, 0.3]))

        spectra = Spectra(
            np.array([1, 2, 3]),
            np.array([2, 3, 4]),
            "nm",
            2.0,
            0.5,
            1.0,
            covariance=np.diag(np.array([0.1, 0.2, 0.3]) ** 2),
        )
        sum_spectra = deepcopy(self.spectra)
        self.spectra._add(spectra, sum_spectra)
        npt.assert_equal(sum_spectra.wavelengths, np.array([1, 2, 3]))
        npt.assert_equal(sum_spectra.flux, np.array([6, 8, 10]))
        npt.assert_equal(
            sum_spectra.covariance, 2 * np.diag(np.array([0.1, 0.2, 0.3]) ** 2)
        )

        sum_spectra = self.spectra + spectra
        npt.assert_equal(sum_spectra.wavelengths, np.array([1, 2, 3]))
        npt.assert_equal(sum_spectra.flux, np.array([6, 8, 10]))

        spectra += self.spectra
        npt.assert_equal(spectra.wavelengths, np.array([1, 2, 3]))
        npt.assert_equal(spectra.flux, np.array([6, 8, 10]))

    def test_concat_function(self):
        """Test the _concat method and concatenation operator of the Spectra object."""
        spectra = Spectra(
            np.array([5, 6, 7]),
            np.array([2, 3, 4]),
            "nm",
            2.0,
            0.5,
            1.0,
            noise=np.array([0.1, 0.2, 0.3]),
        )

        cat_spectra = deepcopy(self.spectra)
        self.spectra._concat(spectra, cat_spectra)
        npt.assert_equal(cat_spectra.wavelengths, np.array([1, 2, 3, 5, 6, 7]))
        npt.assert_equal(cat_spectra.flux, np.array([4, 5, 6, 2, 3, 4]))
        npt.assert_equal(cat_spectra.noise, np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]))

        spectra = Spectra(
            np.array([5, 6, 7]),
            np.array([2, 3, 4]),
            "nm",
            2.0,
            0.5,
            1.0,
            covariance=np.diag(np.array([0.1, 0.2, 0.3]) ** 2),
        )

        cat_spectra = deepcopy(self.spectra)
        self.spectra._concat(spectra, cat_spectra)
        npt.assert_equal(cat_spectra.wavelengths, np.array([1, 2, 3, 5, 6, 7]))
        npt.assert_equal(cat_spectra.flux, np.array([4, 5, 6, 2, 3, 4]))
        npt.assert_equal(
            cat_spectra.covariance,
            np.diag(np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]) ** 2),
        )

        cat_spectra = self.spectra & spectra
        npt.assert_equal(cat_spectra.wavelengths, np.array([1, 2, 3, 5, 6, 7]))
        npt.assert_equal(cat_spectra.flux, np.array([4, 5, 6, 2, 3, 4]))

        self.spectra &= spectra
        npt.assert_equal(self.spectra.wavelengths, np.array([1, 2, 3, 5, 6, 7]))
        npt.assert_equal(self.spectra.flux, np.array([4, 5, 6, 2, 3, 4]))


class TestDatacube:
    """Test suite for the Datacube class."""

    def setup_method(self):
        """Setup method to initialize the Datacube object with test data."""
        self.wavelengths = np.arange(10)
        self.flux = np.random.normal(size=(10, 3, 3))
        self.flux_unit = "arbitrary"
        self.wavelength_unit = "nm"
        self.noise = np.ones_like(self.flux)
        self.covariance = np.ones((10, 10, 3, 3))
        self.z_lens = 0.5
        self.z_source = 1.0
        self.fwhm = 2.0
        self.center_pixel_x = 1
        self.center_pixel_y = 1
        self.coordinate_transform_matrix = np.array([[0.1, 0], [0, 0.1]])

        self.datacube = Datacube(
            self.wavelengths,
            self.flux,
            self.wavelength_unit,
            self.fwhm,
            self.z_lens,
            self.z_source,
            self.center_pixel_x,
            self.center_pixel_y,
            self.coordinate_transform_matrix,
            self.flux_unit,
            self.noise,
            covariance=self.covariance,
        )

    def test_center_pixel_x(self):
        """Test the center_pixel_x property of the Datacube object."""
        assert self.datacube.center_pixel_x == self.center_pixel_x

    def test_center_pixel_y(self):
        """Test the center_pixel_y property of the Datacube object."""
        assert self.datacube.center_pixel_y == self.center_pixel_y

    def test_x_coordinates(self):
        """Test the x_coordinates property of the Datacube object."""
        npt.assert_array_equal(
            self.datacube.x_coordinates,
            [[-0.1, 0, 0.1], [-0.1, 0, 0.1], [-0.1, 0, 0.1]],
        )

    def test_y_coordinates(self):
        """Test the y_coordinates property of the Datacube object."""
        npt.assert_array_equal(
            self.datacube.y_coordinates,
            [[-0.1, -0.1, -0.1], [0, 0, 0], [0.1, 0.1, 0.1]],
        )

    def test_get_1d_spectra(self):
        """Test the get_1d_spectra method of the Datacube object."""
        # Test with mask
        mask = np.zeros((3, 3))
        mask[1:2, 1:2] = 1
        spectra = self.datacube.get_1d_spectra(mask=mask)
        assert isinstance(spectra, Spectra)
        assert spectra.flux.shape == (10,)
        assert spectra.noise.shape == (10,)
        assert spectra.covariance.shape == (10, 10)

        # Test with specific coordinates
        spectra = self.datacube.get_1d_spectra(x=2, y=2)
        assert isinstance(spectra, Spectra)
        assert spectra.flux.shape == (10,)
        assert spectra.noise.shape == (10,)
        assert spectra.covariance.shape == (10, 10)

        # Test with invalid coordinates
        with pytest.raises(ValueError):
            self.datacube.get_1d_spectra(x=3)

        # Test without any parameters
        spectra = self.datacube.get_1d_spectra()
        assert isinstance(spectra, Spectra)
        assert spectra.flux.shape == (10,)
        assert spectra.noise.shape == (10,)
        assert spectra.covariance.shape == (10, 10)


class TestVoronoiBinnedSpectra:
    """Test suite for the VoronoiBinnedSpectra class."""

    def setup_method(self):
        """Setup method to initialize the VoronoiBinnedSpectra object with test data."""
        # Initialize test data for wavelengths, flux, and other properties
        self.wavelengths = np.arange(10)
        self.flux = np.random.normal(size=(10, 3))
        self.flux_unit = "arbitrary"
        self.wavelength_unit = "nm"
        self.noise = np.ones_like(self.flux)
        self.z_lens = 0.5
        self.z_source = 1.0
        self.fwhm = 2.0

        # Create a grid of x and y coordinates
        x = np.array([0, 1, 2, 3])
        xx, yy = np.meshgrid(x, x)
        self.x_coordinates = xx
        self.y_coordinates = yy

        # Define bin numbers and pixel coordinates
        self.bin_numbers = np.array([0, 1, 2, 2])
        self.x_pixels = np.array([0, 1, 2, 3])
        self.y_pixels = np.array([0, 1, 2, 3])

        # Initialize the VoronoiBinnedSpectra object with the test data
        self.voronoi_binned_spectra = VoronoiBinnedSpectra(
            self.wavelengths,
            self.flux,
            self.wavelength_unit,
            self.fwhm,
            self.z_lens,
            self.z_source,
            self.x_coordinates,
            self.y_coordinates,
            self.bin_numbers,
            self.x_pixels,
            self.y_pixels,
            self.flux_unit,
            self.noise,
            bin_center_x=np.mean(self.x_coordinates),
            bin_center_y=np.mean(self.y_coordinates),
            area=np.ones_like(self.x_coordinates),
            snr=np.ones_like(self.x_coordinates),
        )

    def test_x_coordinates(self):
        """Test the x_coordinates property of the VoronoiBinnedSpectra object."""
        npt.assert_array_equal(
            self.voronoi_binned_spectra.x_coordinates, self.x_coordinates
        )

    def test_y_coordinates(self):
        """Test the y_coordinates property of the VoronoiBinnedSpectra object."""
        npt.assert_array_equal(
            self.voronoi_binned_spectra.y_coordinates, self.y_coordinates
        )

    def test_bin_numbers(self):
        """Test the bin_numbers property of the VoronoiBinnedSpectra object."""
        npt.assert_array_equal(
            self.voronoi_binned_spectra.bin_numbers, self.bin_numbers
        )

    def test_x_pixels_of_bins(self):
        """Test the x_pixels_of_bins property of the VoronoiBinnedSpectra object."""
        npt.assert_array_equal(
            self.voronoi_binned_spectra.x_pixels_of_bins, self.x_pixels
        )

    def test_y_pixels_of_bins(self):
        """Test the y_pixels_of_bins property of the VoronoiBinnedSpectra object."""
        npt.assert_array_equal(
            self.voronoi_binned_spectra.y_pixels_of_bins, self.y_pixels
        )

    def test_get_spaxel_map_with_bin_number(self):
        """Test the get_spaxel_map_with_bin_number method of the VoronoiBinnedSpectra
        object."""
        # Generate the spaxel map using the method
        spaxel_map = self.voronoi_binned_spectra.get_spaxel_map_with_bin_number()

        # Create the expected spaxel map for comparison
        test_map = np.zeros_like(self.x_coordinates) - 1
        test_map[0, 0] = 0
        test_map[1, 1] = 1
        test_map[2, 2] = 2
        test_map[3, 3] = 2

        # Assert that the generated spaxel map matches the expected map
        npt.assert_array_equal(spaxel_map, test_map)

    def test_bin_center_x(self):
        """Test the bin_center_x property of the VoronoiBinnedSpectra object."""
        assert self.voronoi_binned_spectra.bin_center_x == np.mean(self.x_coordinates)

    def test_bin_center_y(self):
        """Test the bin_center_y property of the VoronoiBinnedSpectra object."""
        assert self.voronoi_binned_spectra.bin_center_y == np.mean(self.y_coordinates)

    def test_area(self):
        """Test the area property of the VoronoiBinnedSpectra object."""
        assert np.allclose(
            self.voronoi_binned_spectra.area, np.ones_like(self.x_coordinates)
        )

    def test_snr(self):
        """Test the snr property of the VoronoiBinnedSpectra object."""
        assert np.allclose(
            self.voronoi_binned_spectra.snr, np.ones_like(self.x_coordinates)
        )

    def test_get_single_spectra(self):
        """Test the get_single_spectra method of the VoronoiBinnedSpectra object."""
        # Call the method for a specific bin index
        bin_index = 1
        spectra = self.voronoi_binned_spectra.get_single_spectra(bin_index)

        # Assertions to check the output
        assert isinstance(spectra, Spectra)
        assert spectra.flux.shape == (10,)
        assert spectra.noise.shape == (10,)
        assert np.array_equal(spectra.wavelengths, self.wavelengths)
        assert spectra.wavelength_unit == self.wavelength_unit
        assert spectra.fwhm == self.fwhm
        assert spectra.z_lens == self.z_lens
        assert spectra.z_source == self.z_source
        assert spectra.flux_unit == self.flux_unit

        # Create a temporary copy of the VoronoiBinnedSpectra object
        voronoi_binned_spectra_temp = deepcopy(self.voronoi_binned_spectra)
        voronoi_binned_spectra_temp.covariance = np.zeros(
            (
                voronoi_binned_spectra_temp.flux.shape[0],
                voronoi_binned_spectra_temp.flux.shape[0],
                voronoi_binned_spectra_temp.flux.shape[1],
            )
        )

        # Populate the covariance matrix with diagonal noise values
        for i in range(voronoi_binned_spectra_temp.flux.shape[1]):
            voronoi_binned_spectra_temp.covariance[:, :, i] = np.diag(
                voronoi_binned_spectra_temp.noise[:, i] ** 2
            )
        voronoi_binned_spectra_temp.noise = None

        # Call the method again with the modified object
        spectra = voronoi_binned_spectra_temp.get_single_spectra(bin_index)
        assert spectra.covariance.shape == (10, 10)


class TestRadiallyBinnedSpectra:
    """Test suite for the RadiallyBinnedSpectra class."""

    def setup_method(self):
        """Setup method to initialize the RadiallyBinnedSpectra object with test
        data."""
        # Initialize test data for wavelengths, flux, and other properties
        self.wavelengths = np.arange(10)
        self.flux = np.random.normal(size=(10, 3))
        self.flux_unit = "arbitrary"
        self.wavelength_unit = "nm"
        self.noise = np.ones_like(self.flux)
        self.z_lens = 0.5
        self.z_source = 1.0
        self.fwhm = 2.0

        # Define bin radii
        self.bin_radii = np.array([0, 1, 2, 4])

        # Initialize the RadiallyBinnedSpectra object with the test data
        self.radially_binned_spectra = RadiallyBinnedSpectra(
            self.wavelengths,
            self.flux,
            self.wavelength_unit,
            self.fwhm,
            self.z_lens,
            self.z_source,
            self.bin_radii,
            self.flux_unit,
            self.noise,
        )

    def test_bin_radii(self):
        """Test the bin_radii property of the RadiallyBinnedSpectra object."""
        npt.assert_array_equal(self.radially_binned_spectra.bin_radii, self.bin_radii)


if __name__ == "__main__":
    pytest.main()
