import pytest
import numpy as np
import numpy.testing as npt
from copy import deepcopy
from squirrel.data import Spectra
from squirrel.data import Datacube
from squirrel.data import VoronoiBinnedSpectra
from squirrel.data import RadiallyBinnedSpectra


class TestSpectra:
    def setup_method(self):
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
        npt.assert_array_equal(self.spectra.flux, self.flux)

    def test_flux_setter(self):
        self.spectra.flux = np.array([7, 8, 9])
        npt.assert_array_equal(self.spectra.flux, np.array([7, 8, 9]))

    def test_wavelengths(self):
        npt.assert_array_equal(self.spectra.wavelengths, self.wavelengths)

    def test_wavelengths_setter(self):
        self.spectra.wavelengths = np.array([4, 5, 6])
        npt.assert_array_equal(self.spectra.wavelengths, np.array([4, 5, 6]))

    def test_wavelength_unit(self):
        assert self.spectra.wavelength_unit == self.wavelength_unit

    def test_flux_unit(self):
        assert self.spectra.flux_unit == self.flux_unit

    def test_fwhm(self):
        assert self.spectra.fwhm == self.fwhm

    def test_noise(self):
        npt.assert_array_equal(self.spectra.noise, self.noise)

    def test_noise_setter(self):
        self.spectra.noise = np.array([0.4, 0.5, 0.6])
        npt.assert_array_equal(self.spectra.noise, np.array([0.4, 0.5, 0.6]))

    def test_covariance(self):
        npt.assert_array_equal(self.spectra.covariance, self.covariance)

    def test_covariance_setter(self):
        self.spectra.covariance = np.diag([0.4, 0.5, 0.6])
        npt.assert_array_equal(self.spectra.covariance, np.diag([0.4, 0.5, 0.6]))

    def test_z_lens(self):
        assert self.spectra.z_lens == self.z_lens

    def test_z_source(self):
        assert self.spectra.z_source == self.z_source

    def test_spectra_modifications(self):
        assert self.spectra.spectra_modifications == []

    def test_spectra_modifications_setter(self):
        self.spectra.spectra_modifications = "rebinned"
        assert self.spectra.spectra_modifications == "rebinned"

    def test_velocity_scale(self):
        assert self.spectra.velocity_scale is None

    def test_velocity_scale_setter(self):
        self.spectra.velocity_scale = 1.0
        assert self.spectra.velocity_scale == 1.0

    def test_wavelengths_frame(self):
        assert self.spectra.wavelengths_frame == "observed"

    def test_wavelengths_frame_setter(self):
        self.spectra.wavelengths_frame = "rest"
        assert self.spectra.wavelengths_frame == "rest"

    def test_deredshift(self):
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
        self.spectra.clip(wavelength_min=1.5, wavelength_max=2.5)
        npt.assert_equal(self.spectra.wavelengths, np.array([2]))
        npt.assert_equal(self.spectra.flux, np.array([5]))
        npt.assert_equal(self.spectra.noise, np.array([0.2]))

    def test_reset(self):
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
    def setup_method(self):
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
        assert self.datacube.center_pixel_x == self.center_pixel_x

    def test_center_pixel_y(self):
        assert self.datacube.center_pixel_y == self.center_pixel_y

    def test_x_coordinates(self):
        npt.assert_array_equal(
            self.datacube.x_coordinates,
            [[-0.1, 0, 0.1], [-0.1, 0, 0.1], [-0.1, 0, 0.1]],
        )

    def test_y_coordinates(self):
        npt.assert_array_equal(
            self.datacube.y_coordinates,
            [[-0.1, -0.1, -0.1], [0, 0, 0], [0.1, 0.1, 0.1]],
        )

    def test_get_1d_spectra(self):
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
    def setup_method(self):
        self.wavelengths = np.arange(10)
        self.flux = np.random.normal(size=(10, 3))
        self.flux_unit = "arbitrary"
        self.wavelength_unit = "nm"
        self.noise = np.ones_like(self.flux)
        self.z_lens = 0.5
        self.z_source = 1.0
        self.fwhm = 2.0
        x = np.array([0, 1, 2, 3])
        xx, yy = np.meshgrid(x, x)
        self.x_coordinates = xx
        self.y_coordinates = yy
        self.bin_numbers = np.array([0, 1, 2, 2])
        self.x_pixels = np.array([0, 1, 2, 3])
        self.y_pixels = np.array([0, 1, 2, 3])
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
        npt.assert_array_equal(
            self.voronoi_binned_spectra.x_coordinates, self.x_coordinates
        )

    def test_y_coordinates(self):
        npt.assert_array_equal(
            self.voronoi_binned_spectra.y_coordinates, self.y_coordinates
        )

    def test_bin_numbers(self):
        npt.assert_array_equal(
            self.voronoi_binned_spectra.bin_numbers, self.bin_numbers
        )

    def test_x_pixels_of_bins(self):
        npt.assert_array_equal(
            self.voronoi_binned_spectra.x_pixels_of_bins, self.x_pixels
        )

    def test_y_pixels_of_bins(self):
        npt.assert_array_equal(
            self.voronoi_binned_spectra.y_pixels_of_bins, self.y_pixels
        )

    def test_get_spaxel_map_with_bin_number(self):
        spaxel_map = self.voronoi_binned_spectra.get_spaxel_map_with_bin_number()
        test_map = np.zeros_like(self.x_coordinates) - 1
        test_map[0, 0] = 0
        test_map[1, 1] = 1
        test_map[2, 2] = 2
        test_map[3, 3] = 2
        npt.assert_array_equal(spaxel_map, test_map)

    def test_bin_center_x(self):
        # Check the bin_center_x property
        assert self.voronoi_binned_spectra.bin_center_x == np.mean(self.x_coordinates)

    def test_bin_center_y(self):
        # Check the bin_center_y property
        assert self.voronoi_binned_spectra.bin_center_y == np.mean(self.y_coordinates)

    def test_area(self):
        # Check the area property
        assert np.allclose(
            self.voronoi_binned_spectra.area, np.ones_like(self.x_coordinates)
        )

    def test_get_single_spectra(self):
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


class TestRadiallyBinnedSpectra:
    def setup_method(self):
        self.wavelengths = np.arange(10)
        self.flux = np.random.normal(size=(10, 3))
        self.flux_unit = "arbitrary"
        self.wavelength_unit = "nm"
        self.noise = np.ones_like(self.flux)
        self.z_lens = 0.5
        self.z_source = 1.0
        self.fwhm = 2.0
        self.bin_radii = np.array([0, 1, 2, 4])

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
        npt.assert_array_equal(self.radially_binned_spectra.bin_radii, self.bin_radii)


if __name__ == "__main__":
    pytest.main()
