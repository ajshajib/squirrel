import unittest

from squirrel.data import Data
from squirrel.data import Datacube


class TestData(unittest.TestCase):
    def setUp(self):
        self.wavelengths = [1, 2, 3]
        self.spectra = [4, 5, 6]
        self.spectra_unit = "arbitrary unit"
        self.wavelength_unit = "nm"
        self.mask = [True, False, True]
        self.noise = [0.1, 0.2, 0.3]
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
        self.assertEqual(self.data.spectra, self.spectra)

    def test_wavelengths(self):
        self.assertEqual(self.data.wavelengths, self.wavelengths)

    def test_wavelength_unit(self):
        self.assertEqual(self.data.wavelength_unit, self.wavelength_unit)

    def test_spectra_unit(self):
        self.assertEqual(self.data.spectra_unit, self.spectra_unit)

    def test_fwhm(self):
        self.assertEqual(self.data.fwhm, self.fwhm)

    def test_mask(self):
        self.assertEqual(self.data.mask, self.mask)

    def test_noise(self):
        self.assertEqual(self.data.noise, self.noise)

    def test_z_lens(self):
        self.assertEqual(self.data.z_lens, self.z_lens)

    def test_z_source(self):
        self.assertEqual(self.data.z_source, self.z_source)


class TestDatacube(unittest.TestCase):
    def setUp(self):
        self.wavelengths = [1, 2, 3]
        self.spectra = [[4, 5, 6], [7, 8, 9], [10, 11, 12]]
        self.spectra_unit = "arbitrary"
        self.wavelength_unit = "nm"
        self.mask = [[True, False, True], [False, True, False], [True, False, True]]
        self.noise = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        self.z_lens = 0.5
        self.z_source = 1.0
        self.fwhm = 2.0

        self.datacube = Datacube(
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


if __name__ == "__main__":
    unittest.main()
