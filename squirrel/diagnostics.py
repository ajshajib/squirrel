"""This module contains class and functions for diagnostics."""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy.polynomial import legendre
from ppxf.ppxf import ppxf
from ppxf.ppxf_util import convolve_gauss_hermite
from tqdm.notebook import tqdm

from .util import is_positive_definite
from .util import get_nearest_positive_definite_matrix


class Diagnostics(object):
    """This class contains functions to diagnose the performance of the pipeline."""

    @staticmethod
    def get_specific_signal_and_noise(spectra, mask, z_factor=1.0):
        """Get the mean signal per (wavelength unit)^(1/2) and noise of the spectra.

        :param spectra: spectra object
        :type spectra: squirrel.data.Spectra
        :param mask: mask for spectra
        :type mask: numpy.ndarray
        :param z_factor: multiplicative factor for wavelength (e.g., 1 + z), if the SNR
            is computed at a different frame
        :type z_factor: float
        :return: mean signal per (wavelength unit)^(1/2), and noise
        :rtype: float, float
        """
        # Extract the signal from the spectra using the provided mask
        signal_slice = spectra.flux[mask]
        total_signal = np.sum(signal_slice)

        # Calculate the total noise based on the covariance or noise array
        if spectra.covariance is not None:
            covariance_slice = spectra.covariance[mask][:, mask]
            total_noise = np.sqrt(np.sum(covariance_slice))
        else:
            total_noise = np.sqrt(np.sum(spectra.noise[mask] ** 2))

        # Calculate the wavelength range covered by the mask
        wavelength_slice = spectra.wavelengths[mask]
        delta_lambda = wavelength_slice[-1] - wavelength_slice[0]

        return total_signal / np.sqrt(delta_lambda * z_factor), total_noise

    @classmethod
    def get_specific_snr(cls, spectra, mask, z_factor=1.0):
        """Get the mean SNR of the spectra.

        :param spectra: spectra object
        :type spectra: squirrel.data.Spectra
        :param mask: mask for spectra
        :type mask: numpy.ndarray
        :param z_factor: multiplicative factor for wavelength (e.g., 1 + z), if the SNR
            is computed at a different frame
        :type z_factor: float
        :return: mean SNR per (wavelength unit)^(1/2)
        :rtype: float
        """
        # Get the specific signal and noise using the provided mask and z_factor
        signal, noise = cls.get_specific_signal_and_noise(
            spectra, mask, z_factor=z_factor
        )

        return signal / noise

    @classmethod
    def check_bias_vs_snr(
        cls,
        spectra_data,
        template,
        spectra_mask_for_snr=None,
        target_snrs=np.arange(10, 51, 10),
        input_velocity_dispersions=[250],
        template_weight=1.0,
        polynomial_degree=0,
        multiplicative_polynomial_degree=0,
        polynomial_weights=[1.0],
        multiplicative_component=1.0,
        add_component=0.0,
        num_sample=50,
        z_factor=1.0,
        v_systematic=0.0,
        plot=True,
    ):
        """Check the bias in the velocity dispersion measurement as a function of SNR.

        :param spectra_data: data object
        :type spectra_data: squirrel.data.Spectra
        :param template: template object
        :type template: squirrel.template.Template
        :param spectra_mask_for_snr: mask of the spectra to compute SNR from
        :type spectra_mask_for_snr: numpy.ndarray
        :param target_snrs: target SNRs
        :type target_snrs: numpy.ndarray
        :param input_velocity_dispersions: input velocity dispersions
        :type input_velocity_dispersions: numpy.ndarray
        :param template_weight: weight of the template
        :type template_weight: float
        :param polynomial_degree: degree of the polynomial
        :type polynomial_degree: int
        :param multiplicative_polynomial_degree: degree of the multiplicative polynomial for fitting
        :type multiplicative_polynomial_degree: int
        :param polynomial_weights: weights for the additive polynomial to be added to the
            mock spectra
        :type polynomial_weights: numpy.ndarray
        :param multiplicative_component: multiplicative component to be multiplied to the mock spectra
        :type multiplicative_component: np.ndarray or float
        :param add_component: additive component to be added to the mock spectra
        :type add_component: float
        :param num_sample: number of Monte Carlo noise realizations
        :type num_sample: int
        :param z_factor: multiplicative factor for wavelength (e.g., 1 + z), if the SNR
            is computed at a different frame
        :type z_factor: float
        :param v_systematic: systematic velocity, km/s, the `vsyst` parameter in pPXF
        :type v_systematic: float
        :param plot: plot one example simulation for each input velocity dispersion and
            SNR
        :type plot: bool
        :return: recovered values
        :rtype: tuple
        """
        # If no mask is provided, create a default mask that includes all wavelengths
        if spectra_mask_for_snr is None:
            spectra_mask_for_snr = np.ones_like(spectra_data.wavelengths).astype(bool)

        # Ensure that either covariance or noise is provided in the spectra data
        if spectra_data.covariance is None and spectra_data.noise is None:
            raise ValueError("Either covariance or noise must be provided.")

        # Initialize arrays to store recovered values
        recovered_velocities = np.zeros(
            (len(input_velocity_dispersions), len(target_snrs))
        )
        recovered_velocity_uncertainties = np.zeros(
            (len(input_velocity_dispersions), len(target_snrs))
        )
        recovered_velocity_scatters = np.zeros(
            (len(input_velocity_dispersions), len(target_snrs))
        )

        recovered_dispersions = np.zeros(
            (len(input_velocity_dispersions), len(target_snrs))
        )
        recovered_dispersion_uncertainties = np.zeros(
            (len(input_velocity_dispersions), len(target_snrs))
        )
        recovered_dispersion_scatters = np.zeros(
            (len(input_velocity_dispersions), len(target_snrs))
        )

        recovered_snrs = np.zeros((len(input_velocity_dispersions), len(target_snrs)))

        # Loop over each input velocity dispersion
        for i in tqdm(range(len(input_velocity_dispersions)), desc="Input dispersion"):
            input_dispersion = input_velocity_dispersions[i]
            # Create mock spectra by convolving the template with the input dispersion
            mock_flux = cls.make_convolved_spectra(
                template.flux[:, 0],
                input_dispersion,
                spectra_data.velocity_scale,
                int(spectra_data.velocity_scale / template.velocity_scale),
                data_wavelength=spectra_data.wavelengths,
                data_weight=template_weight,
                polynomial_degree=polynomial_degree,
                polynomial_weights=polynomial_weights,
                multiplicative_polynomial=multiplicative_component,
                v_systematic=v_systematic,
            )

            # Add the additive component to the mock flux
            mock_flux += add_component

            # Loop over each target SNR
            for j in tqdm(range(len(target_snrs)), desc="Target SNR"):
                target_snr = target_snrs[j]
                # Initialize arrays to store samples for each Monte Carlo realization
                dispersion_samples = np.zeros(num_sample)
                dispersion_uncertainty_samples = np.zeros(num_sample)
                velocity_samples = np.zeros(num_sample)
                velocity_uncertainty_samples = np.zeros(num_sample)
                snr_samples = np.zeros(num_sample)

                # Perform Monte Carlo realizations
                for k in range(num_sample):
                    data = deepcopy(spectra_data)
                    data.flux = deepcopy(mock_flux)

                    # Calculate the initial specific SNR
                    initial_specific_snr = cls.get_specific_snr(
                        spectra_data, spectra_mask_for_snr, z_factor=z_factor
                    )

                    # Calculate the noise multiplier to achieve the target SNR
                    noise_multiplier = initial_specific_snr / target_snr

                    # Add noise to the mock spectra
                    if data.covariance is not None:
                        data.covariance *= noise_multiplier**2
                        if not is_positive_definite(data.covariance):
                            data.covariance = get_nearest_positive_definite_matrix(
                                data.covariance
                            )
                        noise = np.random.multivariate_normal(
                            data.flux * 0, data.covariance
                        )
                    elif data.noise is not None:
                        data.noise *= noise_multiplier
                        noise = np.random.normal(
                            data.flux * 0, data.noise, size=len(data.flux)
                        )

                    data.flux += noise

                    # Fit the mock spectra using pPXF
                    mock_ppxf_fit = ppxf(
                        templates=template.flux,
                        galaxy=data.flux,
                        noise=deepcopy(
                            data.covariance
                            if data.covariance is not None
                            else data.noise
                        ),  # sending deepcopy just in case, as pPXF may manipulate the noise array/matrix
                        velscale=data.velocity_scale,
                        start=[0, input_dispersion],
                        plot=False,
                        lam=data.wavelengths,
                        degree=polynomial_degree,
                        mdegree=multiplicative_polynomial_degree,
                        vsyst=v_systematic,
                        quiet=True,
                        velscale_ratio=int(
                            data.velocity_scale / template.velocity_scale
                        ),
                    )
                    # Plot the fit for the first realization if plotting is enabled
                    if plot and k == 0:
                        mock_ppxf_fit.plot()
                        plt.title(
                            f"Input dispersion: {input_dispersion} km/s, SNR: {target_snr}"
                        )
                        plt.show()

                    # Store the results of the fit
                    dispersion_samples[k] = mock_ppxf_fit.sol[1]
                    dispersion_uncertainty_samples[k] = mock_ppxf_fit.error[1]
                    velocity_samples[k] = mock_ppxf_fit.sol[0]
                    velocity_uncertainty_samples[k] = mock_ppxf_fit.error[0]
                    snr_samples[k] = cls.get_specific_snr(
                        data, spectra_mask_for_snr, z_factor=z_factor
                    )

                # Calculate the statistics for the recovered values
                mean, uncertainty_mean, scatter = cls.get_stats(
                    velocity_samples, velocity_uncertainty_samples
                )
                recovered_velocities[i, j] = mean
                recovered_velocity_uncertainties[i, j] = uncertainty_mean
                recovered_velocity_scatters[i, j] = scatter

                mean, uncertainty_mean, scatter = cls.get_stats(
                    dispersion_samples, dispersion_uncertainty_samples
                )
                recovered_dispersions[i, j] = mean
                recovered_dispersion_uncertainties[i, j] = uncertainty_mean
                recovered_dispersion_scatters[i, j] = scatter
                recovered_snrs[i, j] = np.mean(snr_samples)

        return (
            recovered_snrs,
            recovered_dispersions,
            recovered_dispersion_uncertainties,
            recovered_dispersion_scatters,
            recovered_velocities,
            recovered_velocity_uncertainties,
            recovered_velocity_scatters,
        )

    @staticmethod
    def get_stats(values, uncertainties, sigma=3):
        """Compute the mean and standard deviation of the array after sigma- clipping.

        :param values: values
        :type arr: numpy.ndarray
        :param uncertainties: uncertainties
        :type uncertainties: numpy.ndarray
        :param sigma: sigma for clipping
        :type sigma: float
        :return: mean, uncertainty of the mean, and scatter
        :rtype: tuple
        """
        # Generate samples from a normal distribution using the values and uncertainties
        samples = np.random.normal(values, uncertainties, size=(1000, len(values)))
        # Calculate the median of the values
        mean = np.median(values)
        # Calculate the uncertainty of the mean using the standard deviation of the medians
        uncertainty_mean = np.std(np.median(samples, axis=1))
        # Calculate the scatter as the standard deviation of the values
        scatter = np.std(values)

        return mean, uncertainty_mean, scatter

    @classmethod
    def plot_bias_vs_snr(
        cls,
        recovered_values,
        input_velocity_dispersions,
        fig_width=10,
        bias_threshold=0.02,
        show_scatter=True,
        show_mean_uncertainty=False,
        errorbar_kwargs_scatter={},
        errorbar_kwargs_mean={},
        **kwargs,
    ):
        """Plot the bias in the velocity dispersion measurement as a function of SNR.

        This function generates plots to visualize the bias in the
        velocity dispersion and velocity measurements as a function of
        Signal-to-Noise Ratio (SNR). It creates subplots for each input
        velocity dispersion and plots the recovered values along with
        their uncertainties and scatters.

        :param recovered_values: recovered values from check_bias_vs_snr
        :type recovered_values: tuple
        :param input_velocity_dispersions: input velocity dispersions
        :type input_velocity_dispersions: numpy.ndarray
        :param fig_width: width of the figure
        :type fig_width: float
        :param bias_threshold: bias threshold line for plotting
        :type bias_threshold: float
        :param show_scatter: show scatter of the points
        :type show_scatter: bool
        :param show_mean_uncertainty: show mean uncertainty of the
            points
        :type show_mean_uncertainty: bool
        :param errorbar_kwargs_scatter: keyword arguments for errorbar
            for scatter
        :type errorbar_kwargs_scatter: dict
        :param errorbar_kwargs_mean: keyword arguments for errorbar for
            mean uncertainty
        :type errorbar_kwargs_mean: dict
        :return: figure and axes
        :rtype: tuple
        """
        # Unpack the recovered values
        (
            recovered_snrs,
            recovered_dispersions,
            recovered_dispersion_uncertainties,
            recovered_dispersion_scatters,
            recovered_velocities,
            recovered_velocity_uncertainties,
            recovered_velocity_scatters,
        ) = recovered_values

        # Create subplots for each input velocity dispersion
        fig, axes = plt.subplots(
            len(input_velocity_dispersions),
            2,
            figsize=(fig_width, fig_width / 8.0 * len(input_velocity_dispersions)),
        )

        # Plot the bias in velocity dispersion
        cls.plot_bias_vs_snr_single(
            axes[:, 0],
            input_velocity_dispersions,
            recovered_snrs,
            recovered_dispersions,
            recovered_dispersion_uncertainties,
            recovered_dispersion_scatters,
            show_scatter=show_scatter,
            show_mean_uncertainty=show_mean_uncertainty,
            bias_threshold=bias_threshold,
            x_label=r"SNR (${\AA}^{-1/2}$)",
            y_label=r"$\sigma$ (km s$^{-1}$)",
            errorbar_kwargs_mean=errorbar_kwargs_mean,
            errorbar_kwargs_scatter=errorbar_kwargs_scatter,
            **kwargs,
        )

        # Plot the bias in velocity
        cls.plot_bias_vs_snr_single(
            axes[:, 1],
            np.zeros(len(recovered_velocities)),
            recovered_snrs,
            recovered_velocities,
            recovered_velocity_uncertainties,
            recovered_velocity_scatters,
            show_scatter=show_scatter,
            show_mean_uncertainty=show_mean_uncertainty,
            bias_threshold=0.0,
            x_label=r"SNR (${\AA}^{-1/2}$)",
            y_label=r"$\Delta v$ (km s$^{-1}$)",
            errorbar_kwargs_mean=errorbar_kwargs_mean,
            errorbar_kwargs_scatter=errorbar_kwargs_scatter,
            **kwargs,
        )

        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.0)

        return fig, axes

    @staticmethod
    def plot_bias_vs_snr_single(
        axes,
        input_values,
        recovered_snrs,
        recovered_values,
        recovered_value_uncertainties,
        recovered_value_scatters,
        show_scatter=True,
        show_mean_uncertainty=False,
        bias_threshold=0.02,
        x_label="",
        y_label="",
        errorbar_kwargs_mean={},
        errorbar_kwargs_scatter={},
        **kwargs,
    ):
        """Plot the bias in velocity dispersion measurement as a function of SNR.

        This function visualizes the bias in a single kinematic value as a function of
        Signal-to-Noise Ratio (SNR). It plots the recovered values along with their
        uncertainties and scatters.

        :param axes: array of ax objects to plot, must match the length of `input_values`
        :type axes: numpy.ndarray or list
        :param input_values: input values
        :type input_values: numpy.ndarray
        :param recovered_snrs: recovered SNRs
        :type recovered_snrs: numpy.ndarray
        :param recovered_values: recovered values
        :type recovered_values: numpy.ndarray
        :param recovered_value_uncertainties: uncertainties of recovered values
        :type recovered_value_uncertainties: numpy.ndarray
        :param recovered_value_scatters: scatters of recovered values
        :type recovered_value_scatters: numpy.ndarray
        :param show_scatter: whether to show scatter of the points
        :type show_scatter: bool
        :param show_mean_uncertainty: whether to show mean uncertainty of the points
        :type show_mean_uncertainty: bool
        :param bias_threshold: bias threshold line for plotting
        :type bias_threshold: float
        :param x_label: label for the x-axis
        :type x_label: str
        :param y_label: label for the y-axis
        :type y_label: str
        :param errorbar_kwargs_mean: keyword arguments for errorbar for mean uncertainty
        :type errorbar_kwargs_mean: dict
        :param errorbar_kwargs_scatter: keyword arguments for errorbar for scatter
        :type errorbar_kwargs_scatter: dict
        :return: None
        :rtype: None
        """

        # Assert axes has the same length as input_values
        assert len(axes) == len(input_values)

        # Plot the recovered values with scatter and mean uncertainty
        for i, input_value in enumerate(input_values):
            if show_scatter:
                axes[i].errorbar(
                    recovered_snrs[i],
                    recovered_values[i],
                    yerr=recovered_value_scatters[i],
                    **errorbar_kwargs_mean,
                )
            if show_mean_uncertainty:
                axes[i].errorbar(
                    recovered_snrs[i],
                    recovered_values[i],
                    yerr=recovered_value_uncertainties[i],
                    **errorbar_kwargs_scatter,
                )

            # Set default marker and linestyle if not provided
            if "marker" not in kwargs:
                kwargs["marker"] = "o"
            if "ls" not in kwargs:
                kwargs["ls"] = "--"
            if "markersize" not in kwargs:
                kwargs["markersize"] = 2

            # Plot the recovered values
            axes[i].plot(
                recovered_snrs[i],
                recovered_values[i],
                mec="k",
                zorder=20,
                **kwargs,
            )
            # Plot the input value as a horizontal line
            axes[i].axhline(input_value, color="grey", linestyle="--", alpha=0.6)

            # Plot the bias threshold as a shaded region
            if bias_threshold > 0.0:
                axes[i].axhspan(
                    input_value * (1 - bias_threshold),
                    input_value * (1 + bias_threshold),
                    color="grey",
                    alpha=0.3,
                    zorder=-10,
                )
            axes[i].set_ylabel(y_label)

        axes[-1].set_xlabel(x_label)

    @classmethod
    def make_convolved_spectra(
        cls,
        template_flux,
        velocity_dispersion,
        velocity_scale,
        velocity_scale_ratio,
        data_wavelength,
        velocity=0.0,
        data_weight=1,
        polynomial_degree=0,
        polynomial_weights=[1.0],
        multiplicative_polynomial=1.0,
        v_systematic=0.0,
    ):
        """Make a convolved spectra.

        This function generates a mock spectrum by convolving a template spectrum with
        a given velocity dispersion. It also applies a polynomial to the convolved
        spectrum to simulate various observational effects.

        :param template_flux: template flux. Wavelengths are not needed as `v_systematic` and `velocity_scale_ratio` will be used to obtain that.
        :type template_flux: numpy.ndarray
        :param velocity_dispersion: velocity dispersion, in km/s
        :type velocity_dispersion: float
        :param velocity_scale: velocity scale, in km/s
        :type velocity_scale: float
        :param velocity_scale_ratio: velocity scale ratio
        :type velocity_scale_ratio: int
        :param data_wavelength: data wavelength
        :type data_wavelength: numpy.ndarray
        :param velocity: velocity, km/s
        :type velocity: float
        :param data_weight: multiplicative factor for the template flux, effective when
            polynomial_degree > 0 to set the relative amplitude of data and polynomial
        :type data_weight: float
        :param polynomial_degree: degree of the polynomial
        :type polynomial_degree: int
        :param polynomial_weights: weights of the polynomial
        :type polynomial_weights: numpy.ndarray
        :param v_systematic: systematic velocity, km/s, the `vsyst` parameter in pPXF
        :type v_systematic: float
        :return: convolved spectra
        :rtype: numpy.ndarray
        """
        # Number of pixels in the data wavelength array
        data_num_pix = len(data_wavelength)

        # Convolve the template flux with the given velocity dispersion
        galaxy_model = convolve_gauss_hermite(
            template_flux,
            velocity_scale,
            np.array([velocity, velocity_dispersion]),
            data_num_pix,
            velscale_ratio=velocity_scale_ratio,
            vsyst=v_systematic,
        )

        # Generate a Legendre polynomial
        x = np.linspace(-1, 1, data_num_pix)
        vand = legendre.legvander(x, polynomial_degree)

        # Return the convolved spectrum with the polynomial applied
        return multiplicative_polynomial * data_weight * galaxy_model + np.dot(
            vand, polynomial_weights
        )
