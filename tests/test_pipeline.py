import pytest
import os
import numpy as np
import numpy.testing as npt
from scipy.special import ndtr
from copy import deepcopy
from squirrel.pipeline import Pipeline
from squirrel.data import Spectra
from squirrel.data import Datacube
from squirrel.template import Template


class MockPpxfFit:
    """
    A mock class to simulate the behavior of a ppxf fit object.
    This class is used for testing purposes.
    """

    def __init__(self):
        self.goodpixels = np.arange(100)
        self.weights = np.linspace(0.2, 1, 10)
        self.degree = 4
        self.sky = np.random.rand(100, 2)
        self.sol = [100.0, 200.0]
        self.mdegree = 2
        self.galaxy = np.random.normal(1, 0.1, 100)
        self.bestfit = np.ones(100) * np.random.normal(1, 0.1)
        self.original_noise = np.diag(np.ones(100) * 0.1**2)


class TestPipeline:
    """
    A test class for the Pipeline module.
    This class contains various test methods to ensure the correctness of the Pipeline functionalities.
    """

    def setup_method(self):
        """
        Set up the test environment.
        This method is called before each test method to set up any state that is shared between tests.
        """
        self.wavelengths = np.arange(9e3, 1.2e4, 5)
        self.flux = np.ones_like(self.wavelengths)
        self.flux_unit = "arbitrary unit"
        self.wavelength_unit = "AA"
        self.noise = np.ones_like(self.flux)
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
            covariance=self.covariance,
        )

    def test_log_rebin(self):
        """
        Test the log_rebin method of the Pipeline class.
        This method tests the log rebinning of spectra and checks for expected modifications.
        """
        # Perform log rebinning on the spectra
        Pipeline.log_rebin(self.spectra, num_samples_for_covariance=10)
        assert "log_rebinned" in self.spectra.spectra_modifications

        # Test for ValueError when log rebinning is performed again
        with pytest.raises(ValueError):
            Pipeline.log_rebin(self.spectra, num_samples_for_covariance=10)

        # Test log rebinning with num_samples_for_covariance as None
        self.spectra.reset()
        Pipeline.log_rebin(self.spectra, num_samples_for_covariance=None)
        assert "log_rebinned" in self.spectra.spectra_modifications

        # Test log rebinning when flux_flattened.shape[0] == 1
        self.spectra.reset()
        self.spectra.flux = self.spectra.flux[np.newaxis, :].T
        Pipeline.log_rebin(self.spectra, num_samples_for_covariance=10)
        assert "log_rebinned" in self.spectra.spectra_modifications

    def test_voronoi_binning(self):
        """
        Test the Voronoi binning functionality of the Pipeline class.
        This method tests the creation of Voronoi bins and the extraction of binned spectra.
        """
        # Create a grid of coordinates
        x = np.arange(11)
        y = np.arange(11)
        xx, yy = np.meshgrid(x, y)

        # Define the center pixel and coordinate transformation matrix
        center_pixel_x = 5
        center_pixel_y = 5
        coordinate_transform_matrix = np.array([[0.1, 0], [0, 0.1]])

        # Calculate the radial distance from the center pixel
        r = np.sqrt((xx - 5) ** 2 + (yy - 5) ** 2)

        # Define the central signal-to-noise ratio (SNR)
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

        # Get the Voronoi binning map
        bin_mapping_output = Pipeline.get_voronoi_binning_map(
            datacube, signal_image, noise_image, 10, max_radius=1.0, plot=True
        )

        # Get the Voronoi binned spectra
        voronoi_binned_spectra = Pipeline.get_voronoi_binned_spectra(
            datacube, bin_mapping_output
        )
        npt.assert_equal(datacube.wavelengths, voronoi_binned_spectra.wavelengths)
        assert datacube.wavelength_unit == voronoi_binned_spectra.wavelength_unit
        assert datacube.fwhm == voronoi_binned_spectra.fwhm
        assert datacube.z_lens == voronoi_binned_spectra.z_lens
        assert datacube.z_source == voronoi_binned_spectra.z_source
        assert datacube.flux_unit == voronoi_binned_spectra.flux_unit
        assert voronoi_binned_spectra.noise.shape == voronoi_binned_spectra.noise.shape
        assert voronoi_binned_spectra.covariance is None
        npt.assert_equal(
            datacube.spectra_modifications, voronoi_binned_spectra.spectra_modifications
        )
        npt.assert_equal(
            datacube.wavelengths_frame, voronoi_binned_spectra.wavelengths_frame
        )

        # Test Voronoi binning with covariance
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
            covariance=np.ones((100, 100, 11, 11)),
        )
        signal_image = np.ones(datacube.flux.shape[1:]) * 9
        noise_image = np.ones_like(signal_image)

        # Get the Voronoi binning map
        bin_mapping_output = Pipeline.get_voronoi_binning_map(
            datacube, signal_image, noise_image, 10, max_radius=1.0, plot=True
        )

        # Get the Voronoi binned spectra
        voronoi_binned_spectra = Pipeline.get_voronoi_binned_spectra(
            datacube, bin_mapping_output
        )
        npt.assert_equal(datacube.wavelengths, voronoi_binned_spectra.wavelengths)
        assert datacube.wavelength_unit == voronoi_binned_spectra.wavelength_unit
        assert datacube.fwhm == voronoi_binned_spectra.fwhm
        assert datacube.z_lens == voronoi_binned_spectra.z_lens
        assert datacube.z_source == voronoi_binned_spectra.z_source
        assert datacube.flux_unit == voronoi_binned_spectra.flux_unit
        assert voronoi_binned_spectra.noise is None
        assert voronoi_binned_spectra.covariance.shape == (
            voronoi_binned_spectra.flux.shape[0],
            voronoi_binned_spectra.flux.shape[0],
            voronoi_binned_spectra.flux.shape[1],
        )
        npt.assert_equal(
            datacube.spectra_modifications, voronoi_binned_spectra.spectra_modifications
        )
        npt.assert_equal(
            datacube.wavelengths_frame, voronoi_binned_spectra.wavelengths_frame
        )

    def test_create_kinematic_map_from_bins(self):
        """
        Test the creation of kinematic maps from bin mappings.
        This method tests the creation of kinematic maps using bin mappings and input values.
        """
        # Define the bin mapping and test map
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

        # Create the kinematic map from bins
        kinematic_map = Pipeline.create_kinematic_map_from_bins(
            bin_mapping, [100, 200, 300]
        )
        npt.assert_equal(kinematic_map, test_map)

    def test_get_template_from_library(self):
        """
        Test the retrieval of templates from a library.
        This method tests the retrieval of templates from a specified library and checks for expected properties.
        """
        library_path = f"{os.path.dirname(__file__)}/spectra_emiles_short_9.0.npz"

        # Test for AssertionError when spectra is not log rebinned
        with pytest.raises(AssertionError):
            Pipeline.get_template_from_library(
                library_path,
                self.spectra,
                2,
            )

        # Perform log rebinning on the spectra
        Pipeline.log_rebin(self.spectra, num_samples_for_covariance=10)
        template = Pipeline.get_template_from_library(
            library_path,
            self.spectra,
            2,
        )

        assert template.flux.shape[1] == 2

    def test_run_ppxf(self):
        """
        Test the pPXF fitting functionality of the Pipeline class.
        This method tests the pPXF fitting on spectra using templates and checks for expected results.
        """
        # Define the wavelength range and line properties
        start_wavelength = 9100
        end_wavelength = 9600
        line_mean = 9350
        line_sigma = 20

        # Create the wavelengths and flux for the spectra
        wavelengths = np.arange(start_wavelength, end_wavelength, 0.5)
        flux = (
            -np.exp(-0.5 * (wavelengths - line_mean) ** 2 / line_sigma**2)
            + (wavelengths - line_mean) / 1000
        )
        noise = np.ones_like(flux) * 0.1
        fwhm = 0.0
        spectra = Spectra(wavelengths, flux, "nm", fwhm, 0.5, 2.0, noise=noise)

        # Create the template wavelengths and fluxes
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

        # Perform log rebinning on the spectra and template
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
        input_velocity_dispersion = line_sigma / line_mean * 299792.458

        # Regular test for pPXF fitting
        ppxf_fit = Pipeline.run_ppxf(spectra, template, start=[0, 600], degree=4)
        assert ppxf_fit.sol[1] == pytest.approx(input_velocity_dispersion, rel=0.005)

        # Test pPXF fitting with tiled flux and noise
        spectra.flux = np.tile(flux, (2, 1)).T
        spectra.noise = np.tile(noise, (2, 1)).T
        spectra.covariance = None
        ppxf_fit = Pipeline.run_ppxf(
            spectra, template, start=[0, 600], degree=4, spectra_indices=0
        )
        assert ppxf_fit.sol[1] == pytest.approx(input_velocity_dispersion, rel=0.005)

        # Test pPXF fitting with no noise
        spectra_temp = deepcopy(spectra)
        spectra_temp.noise = None
        ppxf_fit = Pipeline.run_ppxf(
            spectra_temp, template, start=[0, 600], degree=4, spectra_indices=0
        )
        assert ppxf_fit.sol[1] == pytest.approx(input_velocity_dispersion, rel=0.005)

        # Test pPXF fitting with covariance
        spectra.covariance = np.tile(np.diag(noise**2), (2, 1, 1)).T
        spectra.noise = None
        ppxf_fit = Pipeline.run_ppxf(
            spectra, template, start=[0, 600], degree=4, spectra_indices=0
        )
        assert ppxf_fit.sol[1] == pytest.approx(input_velocity_dispersion, rel=0.005)

        # Test pPXF fitting with non-positive-definite covariance
        spectra.flux = flux
        spectra.covariance = np.diag(noise**2)
        spectra.covariance[0, 0] = 0
        ppxf_fit = Pipeline.run_ppxf(spectra, template, start=[0, 600], degree=4)
        assert ppxf_fit.sol[1] == pytest.approx(input_velocity_dispersion, rel=0.005)

        # Test error raising when providing no index for multidimensional spectra
        spectra.flux = np.tile(flux, (2, 2, 1)).T
        spectra.noise = np.tile(noise, (2, 2, 1)).T
        spectra.covariance = None
        with pytest.raises(ValueError):
            ppxf_fit = Pipeline.run_ppxf(spectra, template, start=[0, 600], degree=4)

        # Test incorrect indexing for 1D arranged spectra (e.g., Voronoi binned)
        spectra.flux = np.tile(flux, (2, 1)).T
        spectra.noise = np.tile(noise, (2, 1)).T
        spectra.covariance = None
        with pytest.raises(ValueError):
            Pipeline.run_ppxf(
                spectra,
                template,
                start=[0, 600],
                degree=4,
                spectra_indices=[0, 0],
            )

        # Test incorrect indexing with datacube
        spectra.flux = np.tile(flux, (2, 2, 1)).T
        spectra.noise = np.tile(noise, (2, 2, 1)).T
        spectra.covariance = None
        ppxf_fit = Pipeline.run_ppxf(
            spectra, template, start=[0, 600], degree=4, spectra_indices=[0, 0]
        )

        assert ppxf_fit.sol[1] == pytest.approx(input_velocity_dispersion, rel=0.005)

        # Test error raising when providing incorrect index for datacube
        with pytest.raises(ValueError):
            Pipeline.run_ppxf(
                spectra,
                template,
                start=[0, 600],
                degree=4,
                spectra_indices=0,
            )

        # Test pPXF fitting with correct indexing for datacube and covariance
        spectra.covariance = np.tile(np.diag(noise**2), (2, 2, 1, 1)).T
        spectra.noise = None
        ppxf_fit = Pipeline.run_ppxf(
            spectra, template, start=[0, 600], degree=4, spectra_indices=[0, 0]
        )
        assert ppxf_fit.sol[1] == pytest.approx(input_velocity_dispersion, rel=0.005)

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
        flux2 = np.random.normal(1, 0.1, len(wavelengths))
        flux3 = np.random.normal(1, 0.1, (len(wavelengths), 2))
        emission_line_groups = [0, 0, 1]

        template1 = Template(wavelengths, flux1, "AA", 2.0)
        template2 = Template(wavelengths, flux2, "AA", 2.0)
        template3 = Template(wavelengths, flux3, "AA", 2.0)

        # Test joining two kinematic templates
        joined_template, component_indices, emission_line_indices = (
            Pipeline.join_templates(template1, template2)
        )
        assert joined_template.flux.shape[1] == 6
        assert np.all(component_indices[: flux1.shape[1]] == 0)
        assert np.all(component_indices[flux1.shape[1] :] == 1)
        for i in emission_line_indices:
            assert i is np.False_

        with pytest.raises(AssertionError):
            template2.flux = np.squeeze(template2.flux)
            joined_template, component_indices, emission_line_indices = (
                Pipeline.join_templates(template2, template1)
            )

        # Test joining two kinematic templates and an emission line template
        flux2 = np.random.normal(1, 0.1, (len(wavelengths), 2))
        template2 = Template(wavelengths, flux2, "AA", 2.0)
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
        fwhm_template = 0.5
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
            2,
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

        # with no weights
        k, n, log_likelihood = Pipeline.get_terms_in_bic(
            ppxf_fit, num_fixed_parameters=1, weight_threshold=None
        )

        # Assertions to check the output
        assert isinstance(k, (int, np.integer))
        assert isinstance(n, (int, np.integer))
        assert isinstance(log_likelihood, (float, np.floating))

        # Check the values
        assert k > 0
        assert n == len(ppxf_fit.goodpixels)

        # with multi-dimensional ppxf.sol
        ppxf_fit.sol = np.array([[100.0, 200.0], [100.0, 200.0]])
        k, n, log_likelihood = Pipeline.get_terms_in_bic(
            ppxf_fit, num_fixed_parameters=1, weight_threshold=None
        )

        # Assertions to check the output
        assert isinstance(k, (int, np.integer))
        assert isinstance(n, (int, np.integer))
        assert isinstance(log_likelihood, (float, np.floating))

        # Check the values
        assert k > 0
        assert n == len(ppxf_fit.goodpixels)

    def test_get_bic(self):
        ppxf_fit = MockPpxfFit()

        # Call the method
        bic = Pipeline.get_bic(ppxf_fit, num_fixed_parameters=1, weight_threshold=0.01)

        # Assertions to check the output
        assert isinstance(bic, float)

        # Check the value
        assert bic > 0

    def test_get_bic_from_sample(self):
        # Create a mock ppxf_fit object

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

    def test_get_relative_bic_weights_for_sample(self):
        # Create mock ppxf fits
        ppxf_fits_list = np.array(
            [[MockPpxfFit(), MockPpxfFit()], [MockPpxfFit(), MockPpxfFit()]]
        )

        # Call the method
        weights = Pipeline.get_relative_bic_weights_for_sample(
            ppxf_fits_list,
            num_fixed_parameters=1,
            num_bootstrap_samples=10,
            weight_threshold=0.01,
        )

        # Assertions to check the output
        assert isinstance(weights, np.ndarray)
        assert weights.shape == (2,)
        # assert np.all(weights >= 0)

    def test_combine_measurements_from_templates(self):
        ppxf_fits_list = np.array(
            [[MockPpxfFit(), MockPpxfFit()], [MockPpxfFit(), MockPpxfFit()]]
        )
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        uncertainties = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Call the method
        (
            combined_values,
            combined_systematic_uncertainty,
            combined_statistical_uncertainty,
            covariance,
        ) = Pipeline.combine_measurements_from_templates(
            values,
            uncertainties,
            ppxf_fits_list,
            apply_bic_weighting=True,
            num_fixed_parameters=0,
            num_bootstrap_samples=10,
            weight_threshold=0.01,
            do_bessel_correction=True,
            verbose=True,
        )

        # Assertions to check the output
        assert isinstance(combined_values, np.ndarray)
        assert isinstance(combined_systematic_uncertainty, np.ndarray)
        assert isinstance(combined_statistical_uncertainty, np.ndarray)
        assert isinstance(covariance, np.ndarray)

        assert combined_values.shape == (2,)
        assert combined_systematic_uncertainty.shape == (2,)
        assert combined_statistical_uncertainty.shape == (2,)

        # With no BIC weighting
        # Call the method
        (
            combined_values,
            combined_systematic_uncertainty,
            combined_statistical_uncertainty,
            covariance,
        ) = Pipeline.combine_measurements_from_templates(
            values,
            uncertainties,
            ppxf_fits_list,
            apply_bic_weighting=False,
            num_fixed_parameters=0,
            num_bootstrap_samples=10,
            weight_threshold=0.01,
            do_bessel_correction=True,
            verbose=True,
        )

        # Assertions to check the output
        assert isinstance(combined_values, np.ndarray)
        assert isinstance(combined_systematic_uncertainty, np.ndarray)
        assert isinstance(combined_statistical_uncertainty, np.ndarray)
        assert isinstance(covariance, np.ndarray)

        assert combined_values.shape == (2,)
        assert combined_systematic_uncertainty.shape == (2,)
        assert combined_statistical_uncertainty.shape == (2,)

    def test_combine_weighted(self):
        # Create mock values, uncertainties, and weights
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        uncertainties = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        weights = np.array([0.2, 0.3, 0.5])

        # Call the method
        (
            combined_values,
            combined_systematic_uncertainty,
            combined_statistical_uncertainty,
            covariance,
        ) = Pipeline.combine_weighted(
            values,
            uncertainties,
            weights,
            do_bessel_correction=True,
        )

        # Assertions to check the output
        assert isinstance(combined_values, np.ndarray)
        assert isinstance(combined_systematic_uncertainty, np.ndarray)
        assert isinstance(combined_statistical_uncertainty, np.ndarray)
        assert isinstance(covariance, np.ndarray)

        assert combined_values.shape == (2,)
        assert combined_systematic_uncertainty.shape == (2,)
        assert combined_statistical_uncertainty.shape == (2,)
        assert covariance.shape == (2, 2)

        # Check the values
        expected_combined_values = np.average(values, axis=0, weights=weights)
        assert np.allclose(combined_values, expected_combined_values, rtol=1e-5)

        expected_combined_statistical_uncertainty = np.sqrt(
            np.sum(weights[:, np.newaxis] * uncertainties**2, axis=0) / np.sum(weights)
        )
        assert np.allclose(
            combined_statistical_uncertainty,
            expected_combined_statistical_uncertainty,
            rtol=1e-5,
        )

        # Check the covariance matrix
        for i in range(covariance.shape[0]):
            for j in range(covariance.shape[0]):
                if i == j:
                    expected_covariance = (
                        np.sum(
                            weights
                            * (values[:, i] - combined_values[i])
                            * (values[:, j] - combined_values[j])
                        )
                        / (np.sum(weights) - np.sum(weights**2) / np.sum(weights))
                        + combined_statistical_uncertainty[i] ** 2
                    )
                else:
                    expected_covariance = np.sum(
                        weights
                        * (values[:, i] - combined_values[i])
                        * (values[:, j] - combined_values[j])
                    ) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights))
                assert np.isclose(covariance[i, j], expected_covariance, rtol=1e-5)

        # for 1D values
        values = np.array([1.0, 2.0, 3.0])
        uncertainties = np.array([0.1, 0.2, 0.3])
        weights = np.array([0.2, 0.3, 0.5])

        # Call the method
        (
            combined_values,
            combined_systematic_uncertainty,
            combined_statistical_uncertainty,
            covariance,
        ) = Pipeline.combine_weighted(
            values,
            uncertainties,
            weights,
            do_bessel_correction=False,
        )

        # Assertions to check the output
        assert isinstance(combined_values, np.ndarray)
        assert isinstance(combined_systematic_uncertainty, np.ndarray)
        assert isinstance(combined_statistical_uncertainty, np.ndarray)
        assert covariance is None

        assert combined_values.shape == (1,)
        assert combined_systematic_uncertainty.shape == (1,)
        assert combined_statistical_uncertainty.shape == (1,)

        # Check the values
        expected_combined_values = np.average(values, weights=weights)
        assert np.allclose(combined_values, expected_combined_values, rtol=1e-5)

        expected_combined_statistical_uncertainty = np.sqrt(
            np.sum(weights * uncertainties**2) / np.sum(weights)
        )
        assert np.allclose(
            combined_statistical_uncertainty,
            expected_combined_statistical_uncertainty,
            rtol=1e-5,
        )

    def test_calculate_weights_from_bic(self):
        # Define test cases
        test_cases = [
            (0.0, 1.0, 1.0),  # delta_bic = 0, sigma_delta_bic = 1
            (2.0, 1.0, 0.5),  # delta_bic = 2, sigma_delta_bic = 1
            (5.0, 2.0, 0.1),  # delta_bic = 5, sigma_delta_bic = 2
            (-2.0, 1.0, 1.0),  # delta_bic = -2, sigma_delta_bic = 1
        ]

        for delta_bic, sigma_delta_bic, expected_weight in test_cases:
            # Call the method
            weight = Pipeline.calculate_weights_from_bic(delta_bic, sigma_delta_bic)

            # Assertions to check the output
            assert isinstance(weight, float)
            assert weight >= 0.0
            assert weight <= 1.0

            # Check the value
            integral_1 = ndtr(-delta_bic / sigma_delta_bic)
            integral_2 = ndtr(delta_bic / sigma_delta_bic - sigma_delta_bic / 2)
            exp_factor = (sigma_delta_bic**2 / 8) - (delta_bic / 2)

            if integral_2 == 0.0:
                integral2_multiplied = 0.0
            else:
                integral2_multiplied = np.exp(exp_factor + np.log(integral_2))

            expected_weight = integral_1 + integral2_multiplied
            assert np.isclose(weight, expected_weight, rtol=1e-5)

    def test_boost_noise(self):
        boost_factor = 2.0
        boosting_mask = np.zeros_like(self.spectra.noise, dtype=bool)
        boosting_mask[100:-100] = True

        # Call the method
        boosted_spectra = Pipeline.boost_noise(
            self.spectra, boost_factor, boosting_mask
        )

        # Assertions to check the output
        assert isinstance(boosted_spectra, Spectra)
        assert boosted_spectra.noise.shape == self.spectra.noise.shape
        assert boosted_spectra.covariance.shape == self.spectra.covariance.shape

        # Check the noise values
        expected_noise = deepcopy(self.spectra.noise)
        expected_noise[boosting_mask] *= boost_factor
        assert np.allclose(boosted_spectra.noise, expected_noise, rtol=1e-5)

        # Check the covariance matrix
        expected_covariance = deepcopy(self.spectra.covariance)
        expected_covariance[boosting_mask] *= boost_factor
        expected_covariance[:, boosting_mask] *= boost_factor
        assert np.allclose(boosted_spectra.covariance, expected_covariance, rtol=1e-5)

        # Call the method without boosting mask
        boosted_spectra = Pipeline.boost_noise(self.spectra, boost_factor, None)

        # Assertions to check the output
        assert isinstance(boosted_spectra, Spectra)
        assert boosted_spectra.noise.shape == self.spectra.noise.shape
        assert boosted_spectra.covariance.shape == self.spectra.covariance.shape

        # Check the noise values
        expected_noise = deepcopy(self.spectra.noise)
        expected_noise *= boost_factor
        assert np.allclose(boosted_spectra.noise, expected_noise, rtol=1e-5)

        # Check the covariance matrix
        expected_covariance = deepcopy(self.spectra.covariance)
        expected_covariance *= boost_factor**2

        assert np.allclose(boosted_spectra.covariance, expected_covariance, rtol=1e-5)


if __name__ == "__main__":
    pytest.main()
