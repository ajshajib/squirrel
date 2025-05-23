import numpy as np
import pytest
import matplotlib.pyplot as plt
from squirrel.data import Spectra
from squirrel.template import Template
from squirrel.diagnostics import Diagnostics


class TestDiagnostics:
    """Test suite for the Diagnostics class in the squirrel package."""

    def test_get_specific_signal_and_noise(self):
        """Test the get_specific_signal_and_noise method of the Diagnostics class.

        This method calculates the signal and noise for a given spectra object, mask,
        and redshift factor.
        """
        # Create mock data for the test
        wavelengths = np.arange(4000, 5000, 0.1)
        flux = np.random.normal(1, 0.1, len(wavelengths))
        noise = np.random.normal(0.1, 0.01, len(wavelengths))
        covariance = np.diag(noise**2)
        mask = (wavelengths > 4100) & (wavelengths < 4900)
        z_factor = 1.0

        # Create a mock Spectra object
        spectra = Spectra(
            wavelengths,
            flux,
            "AA",
            fwhm=2.0,
            noise=noise,
            covariance=covariance,
        )

        # Call the method
        signal, noise = Diagnostics.get_specific_signal_and_noise(
            spectra, mask, z_factor
        )

        # Assertions to check the output
        assert isinstance(signal, float)
        assert isinstance(noise, float)

        # Check the values
        expected_signal = np.sum(flux[mask]) / np.sqrt(
            (wavelengths[mask][-1] - wavelengths[mask][0]) * z_factor
        )
        expected_noise = np.sqrt(np.sum(covariance[mask][:, mask]))
        assert np.isclose(signal, expected_signal, rtol=1e-5)
        assert np.isclose(noise, expected_noise, rtol=1e-5)

    def test_get_specific_snr(self):
        """Test the get_specific_snr method of the Diagnostics class.

        This method calculates the signal-to-noise ratio (SNR) for a given spectra
        object, mask, and redshift factor.
        """
        # Create mock data for the test
        wavelengths = np.arange(4000, 5000, 0.1)
        flux = np.random.normal(1, 0.1, len(wavelengths))
        noise = np.random.normal(0.1, 0.01, len(wavelengths))
        covariance = np.diag(noise**2)
        mask = (wavelengths > 4100) & (wavelengths < 4900)
        z_factor = 1.0

        # Create a mock Spectra object
        spectra = Spectra(
            wavelengths,
            flux,
            "AA",
            fwhm=2.0,
            noise=noise,
            covariance=covariance,
        )

        # Call the method
        snr = Diagnostics.get_specific_snr(spectra, mask, z_factor)

        # Assertions to check the output
        assert isinstance(snr, float)

        # Check the value
        signal, noise = Diagnostics.get_specific_signal_and_noise(
            spectra, mask, z_factor
        )
        expected_snr = signal / noise
        assert np.isclose(snr, expected_snr, rtol=1e-5)

    def test_check_bias_vs_snr(self):
        """Test the check_bias_vs_snr method of the Diagnostics class.

        This method checks the bias versus signal-to-noise ratio (SNR) for a given
        spectra and template object.
        """
        # Create mock data for the test
        wavelengths = 10 ** np.arange(3.81192418, 3.839058575, 0.00024894)
        flux = np.random.normal(1, 0.1, len(wavelengths))
        noise = np.random.normal(0.1, 0.01, len(wavelengths))
        covariance = np.diag(noise**2)
        mask = (wavelengths > 6600) & (wavelengths < 6700)

        # Create a mock Spectra object
        spectra_data = Spectra(
            wavelengths,
            flux,
            "AA",
            fwhm=2.0,
            noise=noise,
            covariance=covariance,
        )
        spectra_data.velocity_scale = 171.8422640494897

        # Create a mock Template object
        template_wavelengths = 10 ** np.arange(3.80993853, 3.841304895, 0.00012447)
        template_flux = np.ones_like(template_wavelengths)
        template = Template(wavelengths, template_flux, "AA", 2.0)
        template.velocity_scale = 85.92113202474485

        # Call the method
        recovered_values = Diagnostics.check_bias_vs_snr(
            spectra_data,
            template,
            spectra_mask_for_snr=mask,
            target_snrs=np.arange(40, 51, 10),
            input_velocity_dispersions=[250],
            template_weight=1.0,
            polynomial_degree=0,
            multiplicative_polynomial_degree=0,
            polynomial_weights=[1.0],
            multiplicative_component=1.0,
            add_component=0.0,
            num_sample=10,  # Reduced for test speed
            z_factor=1.0,
            v_systematic=-1327.7238493696473,
            plot=True,
        )

        # Assertions to check the output
        assert isinstance(recovered_values, tuple)
        assert len(recovered_values) == 7

        for value in recovered_values:
            assert isinstance(value, np.ndarray)

        # Test with positive definite covariance
        spectra_data.covariance = np.diag(noise**2)
        spectra_data.covariance[0, 0] = 0.0
        recovered_values = Diagnostics.check_bias_vs_snr(
            spectra_data,
            template,
            spectra_mask_for_snr=None,
            target_snrs=np.arange(40, 51, 10),
            input_velocity_dispersions=[250],
            template_weight=1.0,
            polynomial_degree=0,
            multiplicative_polynomial_degree=0,
            polynomial_weights=[1.0],
            multiplicative_component=1.0,
            add_component=0.0,
            num_sample=10,  # Reduced for test speed
            z_factor=1.0,
            v_systematic=-1327.7238493696473,
            plot=False,
        )

        # Assertions to check the output
        assert isinstance(recovered_values, tuple)
        assert len(recovered_values) == 7

        for value in recovered_values:
            assert isinstance(value, np.ndarray)

        # Test with no covariance
        spectra_data.covariance = None
        recovered_values = Diagnostics.check_bias_vs_snr(
            spectra_data,
            template,
            spectra_mask_for_snr=None,
            target_snrs=np.arange(40, 51, 10),
            input_velocity_dispersions=[250],
            template_weight=1.0,
            polynomial_degree=0,
            multiplicative_polynomial_degree=0,
            polynomial_weights=[1.0],
            multiplicative_component=1.0,
            add_component=0.0,
            num_sample=10,  # Reduced for test speed
            z_factor=1.0,
            v_systematic=-1327.7238493696473,
            plot=False,
        )

        # Assertions to check the output
        assert isinstance(recovered_values, tuple)
        assert len(recovered_values) == 7

        for value in recovered_values:
            assert isinstance(value, np.ndarray)

        # Test with no noise or covariance
        spectra_data.noise = None

        with pytest.raises(ValueError):
            Diagnostics.check_bias_vs_snr(
                spectra_data,
                template,
                spectra_mask_for_snr=mask,
                target_snrs=np.arange(40, 51, 10),
                input_velocity_dispersions=[250],
                template_weight=1.0,
                polynomial_degree=0,
                multiplicative_polynomial_degree=0,
                polynomial_weights=[1.0],
                multiplicative_component=1.0,
                add_component=0.0,
                num_sample=10,  # Reduced for test speed
                z_factor=1.0,
                v_systematic=-1327.7238493696473,
                plot=True,
            )

    def test_plot_bias_vs_snr(self):
        """Test the plot_bias_vs_snr method of the Diagnostics class.

        This method plots the bias versus signal-to-noise ratio (SNR) for a given set of
        recovered values and input velocity dispersions.
        """
        # Create mock data for the test
        input_velocity_dispersions = np.array([100, 200])
        recovered_snrs = np.array([[10, 20, 30], [10, 20, 30]])
        recovered_dispersions = np.array([[95, 195, 290], [105, 205, 300]])
        recovered_dispersion_uncertainties = np.array([[5, 5, 5], [5, 5, 5]])
        recovered_dispersion_scatters = np.array([[2, 2, 2], [2, 2, 2]])
        recovered_velocities = np.array([[0, 0, 0], [0, 0, 0]])
        recovered_velocity_uncertainties = np.array([[1, 1, 1], [1, 1, 1]])
        recovered_velocity_scatters = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

        recovered_values = (
            recovered_snrs,
            recovered_dispersions,
            recovered_dispersion_uncertainties,
            recovered_dispersion_scatters,
            recovered_velocities,
            recovered_velocity_uncertainties,
            recovered_velocity_scatters,
        )

        # Call the method
        fig, axes = Diagnostics.plot_bias_vs_snr(
            recovered_values,
            input_velocity_dispersions,
            fig_width=10,
            bias_threshold=0.02,
            show_scatter=True,
            show_mean_uncertainty=True,
            errorbar_kwargs_scatter={"fmt": "o", "color": "blue"},
            errorbar_kwargs_mean={"fmt": "s", "color": "red"},
        )

        # Assertions to check the output
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (len(input_velocity_dispersions), 2)

        # Check the content of the plots
        for ax_row in axes:
            for ax in ax_row:
                assert len(ax.lines) > 0

    def test_plot_bias_vs_snr_single(self):
        """Test the plot_bias_vs_snr_single method of the Diagnostics class.

        This method plots the bias versus signal-to-noise ratio (SNR) for a single set
        of recovered values and input values.
        """
        # Create mock data for the test
        input_values = np.array([100, 200])
        recovered_snrs = np.array([[10, 20, 30], [10, 20, 30]])
        recovered_values = np.array([[95, 195, 290], [105, 205, 300]])
        recovered_value_uncertainties = np.array([[5, 5, 5], [5, 5, 5]])
        recovered_value_scatters = np.array([[2, 2, 2], [2, 2, 2]])

        fig, axes = plt.subplots(len(input_values), 1, figsize=(10, 5))

        # Call the method
        Diagnostics.plot_bias_vs_snr_single(
            axes,
            input_values,
            recovered_snrs,
            recovered_values,
            recovered_value_uncertainties,
            recovered_value_scatters,
            show_scatter=True,
            show_mean_uncertainty=True,
            bias_threshold=0.02,
            x_label="SNR",
            y_label="Value",
            errorbar_kwargs_mean={"fmt": "o", "color": "blue"},
            errorbar_kwargs_scatter={"fmt": "s", "color": "red"},
        )

        # Assertions to check the output
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (len(input_values),)

        # Check the content of the plots
        for ax in axes:
            assert len(ax.lines) > 0

    def test_make_convolved_spectra(self):
        """Test the make_convolved_spectra method of the Diagnostics class.

        This method creates a convolved spectra from a given template flux and various
        parameters such as velocity dispersion, velocity scale, and polynomial weights.
        """
        # Create mock data for the test
        template_wavelengths = 10 ** np.arange(3.80993853, 3.841304895, 0.00012447)
        template_flux = np.random.normal(1, 0.1, len(template_wavelengths))
        velocity_dispersion = 200.0
        velocity_scale = 100.0
        velocity_scale_ratio = 2
        data_wavelength = 10 ** np.arange(3.81192418, 3.839058575, 0.00024894)
        velocity = 0.0
        data_weight = 1.0
        polynomial_degree = 2
        polynomial_weights = [1.0, 0.5, 0.2]
        multiplicative_polynomial = 1.0
        v_systematic = 0.0

        # Call the method
        convolved_spectra = Diagnostics.make_convolved_spectra(
            template_flux,
            velocity_dispersion,
            velocity_scale,
            velocity_scale_ratio,
            data_wavelength,
            velocity=velocity,
            data_weight=data_weight,
            polynomial_degree=polynomial_degree,
            polynomial_weights=polynomial_weights,
            multiplicative_polynomial=multiplicative_polynomial,
            v_systematic=v_systematic,
        )

        # Assertions to check the output
        assert isinstance(convolved_spectra, np.ndarray)
        assert convolved_spectra.shape == data_wavelength.shape

        # Check the values
        assert np.all(np.isfinite(convolved_spectra))


if __name__ == "__main__":
    pytest.main()
