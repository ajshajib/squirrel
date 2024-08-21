import pytest
import numpy as np
import numpy.testing as npt
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
        )

    def test_flux(self):
        npt.assert_array_equal(self.spectra.flux, self.flux)

    def test_flux_setter(self):
        self.spectra.flux = np.array([7, 8, 9])
        npt.assert_array_equal(self.spectra.flux, np.array([7, 8, 9]))

    def test_original_flux(self):
        npt.assert_array_equal(self.spectra.original_flux, self.flux)

    def test_wavelengths(self):
        npt.assert_array_equal(self.spectra.wavelengths, self.wavelengths)

    def test_wavelengths_setter(self):
        self.spectra.wavelengths = np.array([4, 5, 6])
        npt.assert_array_equal(self.spectra.wavelengths, np.array([4, 5, 6]))

    def test_original_wavelengths(self):
        npt.assert_array_equal(self.spectra.original_wavelengths, self.wavelengths)

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

    def test_original_noise(self):
        npt.assert_array_equal(self.spectra.original_noise, self.noise)

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

        self.spectra.reset()
        self.spectra.deredshift(target_frame="source")
        npt.assert_equal(self.spectra.wavelengths, np.array([0.5, 1.0, 1.5]))

        self.spectra.reset()
        self.spectra.deredshift(target_frame="lens")
        npt.assert_equal(
            self.spectra.wavelengths, np.array([1, 2, 3]) / (1 + self.z_lens)
        )

        with pytest.raises(ValueError):
            self.spectra.deredshift(target_frame="unknown")

    def test_clip(self):
        self.spectra.clip(wavelength_min=1.5, wavelength_max=2.5)
        npt.assert_equal(self.spectra.wavelengths, np.array([2]))
        npt.assert_equal(self.spectra.flux, np.array([5]))
        npt.assert_equal(self.spectra.noise, np.array([0.2]))

    def test_reset(self):
        self.spectra.wavelengths = None
        self.spectra.flux = None
        self.spectra.spectra_modifications = None
        self.spectra.wavelengths_state = None
        self.spectra.wavelengths_frame = None

        self.spectra.reset()
        npt.assert_equal(self.spectra.wavelengths, [1, 2, 3])
        npt.assert_equal(self.spectra.flux, [4, 5, 6])
        assert self.spectra.spectra_modifications == []
        assert self.spectra.wavelengths_frame == "observed"


class TestDatacube:
    def setup_method(self):
        self.wavelengths = np.arange(10)
        self.flux = np.random.normal(size=(10, 3, 3))
        self.flux_unit = "arbitrary"
        self.wavelength_unit = "nm"
        self.noise = np.ones_like(self.flux)
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

    def test_spatial_extent(self):
        npt.assert_array_almost_equal(
            self.datacube.spatial_extent(pyplot=False),
            [-0.1, 0.1, -0.1, 0.1],
            decimal=8,
        )
        npt.assert_array_almost_equal(
            self.datacube.spatial_extent(pyplot=True),
            [-0.15, 0.15, -0.15, 0.15],
            decimal=8,
        )

        # below we test the other possible orientations of the coordinates grid
        def _get_datacube(coordinate_transform_matrix):
            return Datacube(
                self.wavelengths,
                self.flux,
                self.wavelength_unit,
                self.fwhm,
                self.z_lens,
                self.z_source,
                self.center_pixel_x,
                self.center_pixel_y,
                coordinate_transform_matrix,
                self.flux_unit,
                self.noise,
            )

        datacube = _get_datacube(
            np.array([[-0.1, 0], [0, 0.1]])
        )  # x-axis decreasing towards right
        npt.assert_array_almost_equal(
            datacube.spatial_extent(pyplot=False),
            [0.1, -0.1, -0.1, 0.1],
            decimal=8,
        )
        npt.assert_array_almost_equal(
            datacube.spatial_extent(pyplot=True),
            [0.15, -0.15, -0.15, 0.15],
            decimal=8,
        )
        datacube = _get_datacube(
            np.array([[0.1, 0], [0, -0.1]])
        )  # y-axis decreasing towards up
        npt.assert_array_almost_equal(
            datacube.spatial_extent(pyplot=False),
            [-0.1, 0.1, 0.1, -0.1],
            decimal=8,
        )
        npt.assert_array_almost_equal(
            datacube.spatial_extent(pyplot=True),
            [-0.15, 0.15, 0.15, -0.15],
            decimal=8,
        )


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
        self.x_coordinates = np.array([0, 1, 2])
        self.y_coordinates = np.array([0, 1, 2])
        self.bin_num = np.array([0, 1, 2])
        self.voronoi_binned_spectra = VoronoiBinnedSpectra(
            self.wavelengths,
            self.flux,
            self.wavelength_unit,
            self.fwhm,
            self.z_lens,
            self.z_source,
            self.x_coordinates,
            self.y_coordinates,
            self.bin_num,
            self.flux_unit,
            self.noise,
        )

    def test_x_coordinates(self):
        npt.assert_array_equal(
            self.voronoi_binned_spectra.x_coordinates, self.x_coordinates
        )

    def test_y_coordinates(self):
        npt.assert_array_equal(
            self.voronoi_binned_spectra.y_coordinates, self.y_coordinates
        )

    def test_bin_num(self):
        npt.assert_array_equal(self.voronoi_binned_spectra.bin_num, self.bin_num)


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
