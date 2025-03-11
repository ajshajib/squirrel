import pytest
import numpy as np
import os
import numpy.testing as npt
from squirrel.pipeline import Pipeline
from squirrel.data import Spectra
from squirrel.data import Datacube
from squirrel.template import Template


class TestPipeline:
    def setup_method(self):
        """Set up the test."""
        self.wavelengths = np.arange(
            9e3,
            1.2e4,
            5,
        )
        self.flux = np.ones_like(self.wavelengths)
        self.flux_unit = "arbitrary unit"
        self.wavelength_unit = "AA"
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
        Pipeline.log_rebin(self.spectra, num_samples_for_covariance=10)
        assert "log_rebinned" in self.spectra.spectra_modifications

        with pytest.raises(ValueError):
            Pipeline.log_rebin(self.spectra, num_samples_for_covariance=10)

    def test_voronoi_binning(self):
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
        signal_image = np.ones(datacube.flux.shape[1:]) * 9
        noise_image = np.ones_like(signal_image)

        bin_mapping_output = Pipeline.get_voronoi_binning_map(
            datacube, signal_image, noise_image, 10, max_radius=1.0, plot=True
        )

        voronoi_binned_spectra = Pipeline.get_voronoi_binned_spectra(
            datacube, bin_mapping_output
        )
        npt.assert_equal(datacube.wavelengths, voronoi_binned_spectra.wavelengths)
        assert datacube.wavelength_unit == voronoi_binned_spectra.wavelength_unit
        assert datacube.fwhm == voronoi_binned_spectra.fwhm
        assert datacube.z_lens == voronoi_binned_spectra.z_lens
        assert datacube.z_source == voronoi_binned_spectra.z_source
        assert datacube.flux_unit == voronoi_binned_spectra.flux_unit
        npt.assert_equal(
            datacube.spectra_modifications, voronoi_binned_spectra.spectra_modifications
        )
        npt.assert_equal(
            datacube.wavelengths_frame, voronoi_binned_spectra.wavelengths_frame
        )

    def test_create_kinematic_map_from_bins(self):
        bin_mapping = np.zeros((4, 4)) - 1
        bin_mapping[0, 0] = 0
        bin_mapping[1, 1] = 1
        bin_mapping[2, 2] = 2
        bin_mapping[3, 3] = 2

        test_map = np.zeros_like(bin_mapping)
        test_map[0, 0] = 100
        test_map[1, 1] = 200
        test_map[2, 2] = 300
        test_map[3, 3] = 300

        kinematic_map = Pipeline.create_kinematic_map_from_bins(
            bin_mapping, [100, 200, 300]
        )
        npt.assert_equal(kinematic_map, test_map)

    def test_get_template_from_library(self):
        library_path = f"{os.path.dirname(__file__)}/spectra_emiles_short_9.0.npz"

        with pytest.raises(AssertionError):
            Pipeline.get_template_from_library(
                library_path,
                self.spectra,
                2,
            )

        Pipeline.log_rebin(self.spectra, num_samples_for_covariance=10)
        template = Pipeline.get_template_from_library(
            library_path,
            self.spectra,
            2,
        )

        assert template.flux.shape[1] == 2

    def test_run_ppxf(self):
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
        template.noise = np.ones_like(template.flux) * 0.01

        Pipeline.log_rebin(
            spectra, take_covariance=False, num_samples_for_covariance=10
        )

        velocity_scale_ratio = 2
        Pipeline.log_rebin(
            template,
            velocity_scale=spectra.velocity_scale / velocity_scale_ratio,
            take_covariance=False,
            num_samples_for_covariance=10,
        )
        ppxf_fit = Pipeline.run_ppxf(spectra, template, start=[0, 600], degree=4)

        input_velocity_dispersion = line_sigma / line_mean * 299792.458
        assert ppxf_fit.sol[1] == pytest.approx(input_velocity_dispersion, rel=0.005)

        spectra.flux = np.tile(flux, (2, 1)).T
        spectra.noise = np.tile(noise, (2, 1)).T
        spectra.covariance = None
        ppxf_fit = Pipeline.run_ppxf(
            spectra, template, start=[0, 600], degree=4, spectra_indices=0
        )
        assert ppxf_fit.sol[1] == pytest.approx(input_velocity_dispersion, rel=0.005)

        with pytest.raises(ValueError):
            Pipeline.run_ppxf(
                spectra,
                template,
                start=[0, 600],
                degree=4,
                spectra_indices=[0, 0],
            )

        spectra.flux = np.tile(flux, (2, 2, 1)).T
        spectra.noise = np.tile(noise, (2, 2, 1)).T
        ppxf_fit = Pipeline.run_ppxf(
            spectra, template, start=[0, 600], degree=4, spectra_indices=[0, 0]
        )

        assert ppxf_fit.sol[1] == pytest.approx(input_velocity_dispersion, rel=0.005)

        with pytest.raises(ValueError):
            Pipeline.run_ppxf(
                spectra,
                template,
                start=[0, 600],
                degree=4,
                spectra_indices=0,
            )

    def test_run_ppxf_on_binned_spectra(self):
        start_wavelength = 9100
        end_wavelength = 9600
        line_mean = 9350
        line_sigma = 20

        wavelengths = np.arange(start_wavelength, end_wavelength, 0.5)
        flux = (
            -np.exp(-0.5 * (wavelengths - line_mean) ** 2 / line_sigma**2)
            + (wavelengths - line_mean) / 1000
        )
        flux = np.tile(flux, (2, 1)).T
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
            templates_wavelengths,
            template_fluxes.T,
            "AA",
            template_fwhm,
        )
        template.noise = np.ones_like(template.flux) * 0.01

        Pipeline.log_rebin(
            spectra, take_covariance=False, num_samples_for_covariance=10
        )

        velocity_scale_ratio = 2
        Pipeline.log_rebin(
            template,
            velocity_scale=spectra.velocity_scale / velocity_scale_ratio,
            take_covariance=False,
            num_samples_for_covariance=10,
        )

        (
            velocity_dispersion,
            velocity_dispersion_uncertainty,
            mean_velocities,
            mean_velocity_uncertainties,
        ) = Pipeline.run_ppxf_on_binned_spectra(
            spectra, template, start=[0, 600], degree=4
        )

        input_velocity_dispersion = line_sigma / line_mean * 299792.458

        npt.assert_allclose(
            velocity_dispersion,
            [input_velocity_dispersion, input_velocity_dispersion],
            rtol=0.01,
            atol=1,
        )

    def test_get_emission_line_template(self):
        # Create mock data for the test
        wavelengths = np.arange(4000, 5000, 0.1)
        fwhm = 2.0
        wavelength_factor = 1.0
        wavelength_range_extend_factor = 1.05

        # Create a mock Spectra object
        spectra_wavelengths = np.arange(4100, 4900, 0.1)
        spectra_flux = np.random.normal(1, 0.1, len(spectra_wavelengths))
        spectra_noise = np.random.normal(0.1, 0.01, len(spectra_wavelengths))
        spectra = Spectra(
            spectra_wavelengths,
            spectra_flux,
            "AA",
            fwhm,
            0.5,
            1.0,
            noise=spectra_noise,
        )
        spectra.spectra_modifications = ["log_rebinned"]

        # Call the method
        template, line_names, line_wavelengths = Pipeline.get_emission_line_template(
            spectra,
            wavelengths,
            wavelength_factor,
            wavelength_range_extend_factor,
        )

        # Assertions to check the output
        assert isinstance(template, Template)
        assert template.wavelength_unit == "AA"
        assert template.fwhm == spectra.fwhm
        assert template.flux.shape[0] == len(wavelengths)
        assert isinstance(line_names, np.ndarray)
        assert isinstance(line_wavelengths, np.ndarray)
        assert len(line_names) == len(line_wavelengths)

    def test_join_templates(self):
        # Create mock templates
        wavelengths = np.arange(4000, 5000, 0.1)
        flux1 = np.random.normal(1, 0.1, (len(wavelengths), 5))
        flux2 = np.random.normal(1, 0.1, (len(wavelengths), 3))
        flux3 = np.random.normal(1, 0.1, (len(wavelengths), 2))
        emission_line_groups = [0, 0, 1]

        template1 = Template(wavelengths, flux1, "AA", 2.0)
        template2 = Template(wavelengths, flux2, "AA", 2.0)
        template3 = Template(wavelengths, flux3, "AA", 2.0)

        # Test joining two kinematic templates
        joined_template, component_indices, emission_line_indices = (
            Pipeline.join_templates(template1, template2)
        )
        assert joined_template.flux.shape[1] == flux1.shape[1] + flux2.shape[1]
        assert np.all(component_indices[: flux1.shape[1]] == 0)
        assert np.all(component_indices[flux1.shape[1] :] == 1)
        for i in emission_line_indices:
            assert i is np.False_

        # Test joining two kinematic templates and an emission line template
        joined_template, component_indices, emission_line_indices = (
            Pipeline.join_templates(
                template1, template2, template3, emission_line_groups
            )
        )
        assert (
            joined_template.flux.shape[1]
            == flux1.shape[1] + flux2.shape[1] + flux3.shape[1]
        )
        assert np.all(component_indices[: flux1.shape[1]] == 0)
        assert np.all(
            component_indices[flux1.shape[1] : flux1.shape[1] + flux2.shape[1]] == 1
        )
        assert np.all(
            component_indices[flux1.shape[1] + flux2.shape[1] :]
            == np.array(emission_line_groups) + 2
        )
        # assert np.all(emission_line_indices[flux1.shape[1] + flux2.shape[1] :] is True)
        for i in range(flux1.shape[1] + flux2.shape[1]):
            assert emission_line_indices[i] is np.False_
        for i in range(flux1.shape[1] + flux2.shape[1], joined_template.flux.shape[1]):
            assert emission_line_indices[i] is np.True_

        # Test joining one kinematic template and an emission line template
        joined_template, component_indices, emission_line_indices = (
            Pipeline.join_templates(
                template1,
                emission_line_template=template3,
                emission_line_groups=emission_line_groups,
            )
        )
        assert joined_template.flux.shape[1] == flux1.shape[1] + flux3.shape[1]
        assert np.all(component_indices[: flux1.shape[1]] == 0)
        assert np.all(
            component_indices[flux1.shape[1] :] == np.array(emission_line_groups) + 1
        )
        for i in range(flux1.shape[1]):
            assert emission_line_indices[i] is np.False_
        for i in range(flux1.shape[1], joined_template.flux.shape[1]):
            assert emission_line_indices[i] is np.True_

    def test_make_template_from_array(self):
        # Create mock data for the test
        wavelengths = np.arange(4000, 5000, 0.1)
        fluxes = np.random.normal(1, 0.1, (len(wavelengths), 5))
        fwhm_template = 2.0
        velocity_scale_ratio = 2.0
        wavelength_factor = 1.0
        wavelength_range_extend_factor = 1.05

        # Create a mock Spectra object
        spectra_wavelengths = np.arange(4100, 4900, 0.1)
        spectra_flux = np.random.normal(1, 0.1, len(spectra_wavelengths))
        spectra_noise = np.random.normal(0.1, 0.01, len(spectra_wavelengths))
        spectra = Spectra(
            spectra_wavelengths,
            spectra_flux,
            "AA",
            fwhm_template,
            0.5,
            1.0,
            noise=spectra_noise,
        )
        spectra.spectra_modifications = ["log_rebinned"]
        spectra.velocity_scale = 100.0  # Set a mock velocity scale

        # Call the method
        template = Pipeline.make_template_from_array(
            fluxes,
            wavelengths,
            fwhm_template,
            spectra,
            velocity_scale_ratio,
            wavelength_factor,
            wavelength_range_extend_factor,
        )

        # Assertions to check the output
        assert isinstance(template, Template)
        assert template.wavelength_unit == "AA"
        assert template.fwhm == spectra.fwhm
        assert template.flux.shape[1] == fluxes.shape[1]
        assert np.all(
            template.wavelengths
            >= spectra.wavelengths[0]
            / wavelength_range_extend_factor
            * wavelength_factor
        )
        assert np.all(
            template.wavelengths
            <= spectra.wavelengths[-1]
            * wavelength_range_extend_factor
            * wavelength_factor
        )

    def test_get_terms_in_bic(self):
        # Create a mock ppxf_fit object
        class MockPpxfFit:
            def __init__(self):
                self.goodpixels = np.arange(100)
                self.weights = np.random.rand(50)
                self.degree = 4
                self.sky = np.random.rand(100, 2)
                self.sol = [100.0, 200.0]
                self.mdegree = 2
                self.galaxy = np.random.rand(100)
                self.bestfit = np.random.rand(100)
                self.original_noise = np.random.rand(100, 100)

        ppxf_fit = MockPpxfFit()

        # Call the method
        k, n, log_likelihood = Pipeline.get_terms_in_bic(
            ppxf_fit, num_fixed_parameters=1, weight_threshold=0.01
        )

        # Assertions to check the output
        assert isinstance(k, (int, np.integer))
        assert isinstance(n, (int, np.integer))
        assert isinstance(log_likelihood, (float, np.floating))

        # Check the values
        assert k > 0
        assert n == len(ppxf_fit.goodpixels)

    def test_get_bic(self):
        # Create a mock ppxf_fit object
        class MockPpxfFit:
            def __init__(self):
                self.goodpixels = np.arange(100)
                self.weights = np.random.rand(50)
                self.degree = 4
                self.sky = np.random.rand(100, 2)
                self.sol = [100.0, 200.0]
                self.mdegree = 2
                self.galaxy = np.random.rand(100)
                self.bestfit = np.random.rand(100)
                self.original_noise = np.random.rand(100, 100)

        ppxf_fit = MockPpxfFit()

        # Call the method
        bic = Pipeline.get_bic(ppxf_fit, num_fixed_parameters=1, weight_threshold=0.01)

        # Assertions to check the output
        assert isinstance(bic, float)

        # Check the value
        assert bic > 0

    def test_get_bic_from_sample(self):
        # Create a mock ppxf_fit object
        class MockPpxfFit:
            def __init__(self):
                self.goodpixels = np.arange(100)
                self.weights = np.random.rand(50)
                self.degree = 4
                self.sky = np.random.rand(100, 2)
                self.sol = [100.0, 200.0]
                self.mdegree = 2
                self.galaxy = np.random.rand(100)
                self.bestfit = np.random.rand(100)
                self.original_noise = np.random.rand(100, 100)

        # Create a list of mock ppxf_fit objects
        ppxf_fits = [MockPpxfFit() for _ in range(5)]

        # Call the method
        bic = Pipeline.get_bic_from_sample(
            ppxf_fits, num_fixed_parameters=1, weight_threshold=0.01
        )

        # Assertions to check the output
        assert isinstance(bic, float)

        # Check the value
        assert bic > 0


if __name__ == "__main__":
    pytest.main()
