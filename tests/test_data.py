import unittest

from squirrel.data import Data
from squirrel.data import Datacube


class TestData(unittest.TestCase):
    def setUp(self):
        self.wavelengths = [1, 2, 3]
        self.spectra = [4, 5, 6]
        self.spectra_unit = "unit"
        self.mask = [True, False, True]
        self.noise = [0.1, 0.2, 0.3]
        self.data = Data(
            self.wavelengths, self.spectra, self.spectra_unit, self.mask, self.noise
        )

    def test_spectra(self):
        self.assertEqual(self.data.spectra, self.spectra)

    def test_wavelengths(self):
        self.assertEqual(self.data.wavelengths, self.wavelengths)

    def test_spectra_unit(self):
        self.assertEqual(self.data.spectra_unit, self.spectra_unit)

    def test_mask(self):
        self.assertEqual(self.data.mask, self.mask)

    def test_noise(self):
        self.assertEqual(self.data.noise, self.noise)


class TestDatacube(unittest.TestCase):
    def setUp(self):
        self.wavelengths = [1, 2, 3]
        self.spectra = [[4, 5, 6], [7, 8, 9], [10, 11, 12]]
        self.spectra_unit = "unit"
        self.mask = [[True, False, True], [False, True, False], [True, False, True]]
        self.noise = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        self.datacube = Datacube(
            self.wavelengths, self.spectra, self.spectra_unit, self.mask, self.noise
        )

    def test_spectra(self):
        self.assertEqual(self.datacube.spectra, self.spectra)

    def test_wavelengths(self):
        self.assertEqual(self.datacube.wavelengths, self.wavelengths)

    def test_spectra_unit(self):
        self.assertEqual(self.datacube.spectra_unit, self.spectra_unit)

    def test_mask(self):
        self.assertEqual(self.datacube.mask, self.mask)

    def test_noise(self):
        self.assertEqual(self.datacube.noise, self.noise)


if __name__ == "__main__":
    unittest.main()
