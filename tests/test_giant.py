import unittest
import numpy as np
from squirrel import giant
from squirrel.data import Spectra
from squirrel.template import Template


class TestShoulder(unittest.TestCase):
    def setUp(self):
        """Set up the test."""
        self.wavelengths = np.arange(1, 100)
        self.flux = np.ones_like(self.wavelengths)
        self.flux_unit = "arbitrary unit"
        self.wavelength_unit = "nm"
        self.noise = np.ones_like(self.flux)
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

    def test_log_rebin(self):
        giant.Shoulder.log_rebin(self.spectra)
        self.assertIn("log_rebinned", self.spectra.spectra_modifications)

    def test_stand(self):
        start_wavelength = 9100
        end_wavelength = 9600
        line_mean = 9350
        line_sigma = 20

        wavelengths = np.arange(start_wavelength, end_wavelength, 0.5)
        flux = (
            -np.exp(-0.5 * (wavelengths - line_mean) ** 2 / line_sigma**2)
            + (wavelengths - line_mean) / 1000
        )
        fwhm = 0.0
        spectra = Spectra(
            wavelengths, flux, "nm", fwhm, 0.5, 2.0, noise=np.zeros_like(flux) + 0.1
        )

        template_sigma = 1
        template_fwhm = 2.355 * template_sigma
        templates_wavelengths = np.arange(
            start_wavelength / 1.2, end_wavelength * 1.2, 0.25
        )
        template_fluxes = np.random.normal(0, 0.1, (10, len(templates_wavelengths)))
        template_fluxes[0] = -np.exp(
            -0.5 * (templates_wavelengths - 9350) ** 2 / template_sigma**2
        )
        template_fluxes[1] = -np.exp(
            -0.5 * (templates_wavelengths - 9450) ** 2 / template_sigma**2
        )

        template = Template(
            templates_wavelengths, template_fluxes.T, "AA", template_fwhm
        )

        giant.Shoulder.log_rebin(spectra)

        velocity_scale_ratio = 2
        giant.Shoulder.log_rebin(
            template, velocity_scale=spectra.velocity_scale / velocity_scale_ratio
        )

        ppxf_fit = giant.Shoulder.stand_on(spectra, template, degree=4)

        input_velocity_dispersion = line_sigma / line_mean * 299792.458
        self.assertAlmostEqual(
            ppxf_fit.sol[1],
            input_velocity_dispersion,
            delta=0.005 * input_velocity_dispersion,
        )


if __name__ == "__main__":
    unittest.main()
