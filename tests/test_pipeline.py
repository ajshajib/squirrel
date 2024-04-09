import unittest
import numpy as np
import numpy.testing as npt
from squirrel.pipeline import Pipeline
from squirrel.data import Spectra
from squirrel.data import Datacube
from squirrel.template import Template


class TestPipeline(unittest.TestCase):
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
        Pipeline.log_rebin(self.spectra)
        self.assertIn("log_rebinned", self.spectra.spectra_modifications)

    def test_voronoi_bin(self):
        x = np.arange(11)
        y = np.arange(11)
        xx, yy = np.meshgrid(x, y)

        center_pixel_x = 5
        center_pixel_y = 5
        coordinate_transform_matrix = np.array([[0.1, 0], [0, 0.1]])

        r = np.sqrt((xx - 5) ** 2 + (yy - 5) ** 2)

        central_snr = 30
        flux = np.ones((100, 11, 11))
        flux *= (central_snr**2 / (1 + r))[np.newaxis, :, :]
        noise = np.sqrt(flux)
        wavelengths = np.arange(900.0, 1000.0, 1.0)
        datacube = Datacube(
            wavelengths,
            flux,
            "nm",
            2.0,
            0.5,
            1.0,
            center_pixel_x,
            center_pixel_y,
            coordinate_transform_matrix,
            noise=noise,
        )
        voronoi_binned_spectra = Pipeline.voronoi_bin(
            datacube, central_snr, 950, 990, 1.0, plot=True
        )
        npt.assert_equal(datacube.wavelengths, voronoi_binned_spectra.wavelengths)
        self.assertEqual(
            datacube.wavelength_unit, voronoi_binned_spectra.wavelength_unit
        )
        self.assertEqual(datacube.fwhm, voronoi_binned_spectra.fwhm)
        self.assertEqual(datacube.z_lens, voronoi_binned_spectra.z_lens)
        self.assertEqual(datacube.z_source, voronoi_binned_spectra.z_source)
        self.assertEqual(datacube.flux_unit, voronoi_binned_spectra.flux_unit)
        npt.assert_equal(
            datacube.spectra_modifications, voronoi_binned_spectra.spectra_modifications
        )
        npt.assert_equal(
            datacube.wavelengths_frame, voronoi_binned_spectra.wavelengths_frame
        )

    def test_stand_on(self):
        start_wavelength = 9100
        end_wavelength = 9600
        line_mean = 9350
        line_sigma = 20

        wavelengths = np.arange(start_wavelength, end_wavelength, 0.5)
        flux = (
            -np.exp(-0.5 * (wavelengths - line_mean) ** 2 / line_sigma**2)
            + (wavelengths - line_mean) / 1000
        )
        noise = np.ones_like(flux) * 0.1
        fwhm = 0.0
        spectra = Spectra(wavelengths, flux, "nm", fwhm, 0.5, 2.0, noise=noise)

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

        Pipeline.log_rebin(spectra)

        velocity_scale_ratio = 2
        Pipeline.log_rebin(
            template, velocity_scale=spectra.velocity_scale / velocity_scale_ratio
        )

        ppxf_fit = Pipeline.run_ppxf(spectra, template, degree=4)

        input_velocity_dispersion = line_sigma / line_mean * 299792.458
        self.assertAlmostEqual(
            ppxf_fit.sol[1],
            input_velocity_dispersion,
            delta=0.005 * input_velocity_dispersion,
        )

        spectra.flux = np.tile(flux, (2, 1)).T
        spectra.noise = np.tile(noise, (2, 1)).T
        ppxf_fit = Pipeline.run_ppxf(spectra, template, degree=4, spectra_indices=0)
        self.assertAlmostEqual(
            ppxf_fit.sol[1],
            input_velocity_dispersion,
            delta=0.005 * input_velocity_dispersion,
        )
        self.assertRaises(
            ValueError,
            Pipeline.run_ppxf,
            spectra,
            template,
            degree=4,
            spectra_indices=[0, 0],
        )

        spectra.flux = np.tile(flux, (2, 2, 1)).T
        spectra.noise = np.tile(noise, (2, 2, 1)).T
        ppxf_fit = Pipeline.run_ppxf(
            spectra, template, degree=4, spectra_indices=[0, 0]
        )
        self.assertAlmostEqual(
            ppxf_fit.sol[1],
            input_velocity_dispersion,
            delta=0.005 * input_velocity_dispersion,
        )

        self.assertRaises(
            ValueError,
            Pipeline.run_ppxf,
            spectra,
            template,
            degree=4,
            spectra_indices=0,
        )


if __name__ == "__main__":
    unittest.main()
