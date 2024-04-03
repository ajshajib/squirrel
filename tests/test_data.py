import unittest
import numpy as np
import numpy.testing as npt
from squirrel.data import Data
from squirrel.data import Datacube


class TestData(unittest.TestCase):
    def setUp(self):
        self.wavelengths = np.array([1, 2, 3])
        self.spectra = np.array([4, 5, 6])
        self.spectra_unit = "arbitrary unit"
        self.wavelength_unit = "nm"
        self.mask = [True, False, True]
        self.noise = np.array([0.1, 0.2, 0.3])
        self.fwhm = 2.0
        self.z_lens = 0.5
        self.z_source = 1.0
        self.data = Data(
            self.wavelengths,
            self.spectra,
            self.wavelength_unit,
            self.fwhm,
            self.z_lens,
            self.z_source,
            self.spectra_unit,
            self.mask,
            self.noise,
        )

    def test_spectra(self):
        npt.assert_array_equal(self.data.spectra, self.spectra)

    def test_spectra_setter(self):
        self.data.spectra = np.array([7, 8, 9])
        npt.assert_array_equal(self.data.spectra, np.array([7, 8, 9]))

    def test_wavelengths(self):
        npt.assert_array_equal(self.data.wavelengths, self.wavelengths)

    def test_wavelengths_setter(self):
        self.data.wavelengths = np.array([4, 5, 6])
        npt.assert_array_equal(self.data.wavelengths, np.array([4, 5, 6]))

    def test_wavelength_unit(self):
        self.assertEqual(self.data.wavelength_unit, self.wavelength_unit)

    def test_spectra_unit(self):
        self.assertEqual(self.data.spectra_unit, self.spectra_unit)

    def test_fwhm(self):
        self.assertEqual(self.data.fwhm, self.fwhm)

    def test_mask(self):
        npt.assert_array_equal(self.data.mask, self.mask)

    def test_noise(self):
        npt.assert_array_equal(self.data.noise, self.noise)

    def test_z_lens(self):
        self.assertEqual(self.data.z_lens, self.z_lens)

    def test_z_source(self):
        self.assertEqual(self.data.z_source, self.z_source)

    def test_spectra_modifications(self):
        self.assertEqual(self.data.spectra_modifications, [])

    def test_spectra_modifications_setter(self):
        self.data.spectra_modifications = "rebinned"
        self.assertEqual(self.data.spectra_modifications, "rebinned")

    def test_velocity_scale(self):
        self.assertIsNone(self.data.velocity_scale)

    def test_velocity_scale_setter(self):
        self.data.velocity_scale = 1.0
        self.assertEqual(self.data.velocity_scale, 1.0)

    def test_wavelengths_frame(self):
        self.assertEqual(self.data.wavelengths_frame, "observed")

    def test_wavelengths_frame_setter(self):
        self.data.wavelengths_frame = "rest"
        self.assertEqual(self.data.wavelengths_frame, "rest")

    def test_deredshift(self):
        self.data.deredshift(redshift=1.0)
        npt.assert_equal(self.data.wavelengths, np.array([0.5, 1.0, 1.5]))

        self.data.reset()
        self.data.deredshift(target_frame="source")
        npt.assert_equal(self.data.wavelengths, np.array([0.5, 1.0, 1.5]))

        self.data.reset()
        self.data.deredshift(target_frame="lens")
        npt.assert_equal(self.data.wavelengths, np.array([1, 2, 3]) / (1 + self.z_lens))

        with self.assertRaises(ValueError):
            self.data.deredshift(target_frame="unknown")

    def test_reset(self):
        self.data.wavelengths = None
        self.data.spectra = None
        self.data.spectra_modifications = None
        self.data.wavelengths_state = None
        self.data.wavelengths_frame = None

        self.data.reset()
        npt.assert_equal(self.data.wavelengths, [1, 2, 3])
        npt.assert_equal(self.data.spectra, [4, 5, 6])
        self.assertEqual(self.data.spectra_modifications, [])
        self.assertEqual(self.data.wavelengths_frame, "observed")


class TestDatacube(unittest.TestCase):
    def setUp(self):
        self.wavelengths = np.arange
        self.spectra = np.random.normal(size=(10, 3, 3))
        self.spectra_unit = "arbitrary"
        self.wavelength_unit = "nm"
        self.mask = np.ones_like(self.spectra, dtype=bool)
        self.noise = np.ones_like(self.spectra)
        self.z_lens = 0.5
        self.z_source = 1.0
        self.fwhm = 2.0
        self.center_pixel_x = 1
        self.center_pixel_y = 1
        self.coordinate_transform_matrix = np.array([[0.1, 0], [0, 0.1]])

        self.datacube = Datacube(
            self.wavelengths,
            self.spectra,
            self.wavelength_unit,
            self.fwhm,
            self.z_lens,
            self.z_source,
            self.center_pixel_x,
            self.center_pixel_y,
            self.coordinate_transform_matrix,
            self.spectra_unit,
            self.mask,
            self.noise,
        )

    def test_center_pixel_x(self):
        self.assertEqual(self.datacube.center_pixel_x, self.center_pixel_x)

    def test_center_pixel_y(self):
        self.assertEqual(self.datacube.center_pixel_y, self.center_pixel_y)

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


if __name__ == "__main__":
    unittest.main()
