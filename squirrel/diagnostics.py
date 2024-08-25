"""This module contains class and functions for diagnostics."""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy.polynomial import legendre
from ppxf.ppxf import losvd_rfft
from ppxf.ppxf import rebin
from tqdm.notebook import tqdm

from .pipeline import Pipeline
from .template import Template


class Diagnostics(object):
    """This class contains functions to diagnose the performance of the pipeline."""

    VEL_LIM = 100  # km/s, limit for velocity bound

    @classmethod
    def check_bias_vs_snr_from_ppxf_fit(
        cls,
        ppxf_fit,
        spectra_data,
        spectra_mask_for_snr=None,
        target_snrs=np.arange(10, 31, 10),
        input_velocity_dispersions=[250],  # np.arange(100, 351, 50)
        add_component=0.0,
        num_samples=50,
        z_factor=1.0,
    ):
        """Check the bias in the velocity dispersion measurement as a function of SNR.

        :param ppxf_fit: ppxf fit object
        :type ppxf_fit: ppxf.ppxf
        :param spectra_data: data object
        :type spectra_data: squirrel.data.Spectra
        :param spectra_mask_for_snr: mask of the spectra to compute SNR from
        :type spectra_mask_for_snr: numpy.ndarray
        :param target_snrs: target SNRs
        :type target_snrs: numpy.ndarray
        :param input_velocity_dispersions: input velocity dispersions
        :type input_velocity_dispersions: numpy.ndarray
        :param add_component: additional component to add to the spectra
        :type add_component: numpy.ndarray
        :param num_samples: number of Monte Carlo noise realizations
        :type num_samples: int
        :param z_factor: multiplicative factor for wavelength (e.g., 1 + z), if the SNR
            is computed at a different frame
        :type z_factor: float
        :return: recovered values
        :rtype: tuple
        """
        template_flux = ppxf_fit.templates[:, 0]
        template_wavelengths = ppxf_fit.lam_temp
        template = Template(template_wavelengths, template_flux, "", 0.0)
        template.velocity_scale = ppxf_fit.velscale / ppxf_fit.velscale_ratio

        template_weight = ppxf_fit.weights[0]
        degree = ppxf_fit.degree
        polynomial_weights = ppxf_fit.polyweights

        return cls.check_bias_vs_snr(
            spectra_data,
            template,
            spectra_mask_for_snr=spectra_mask_for_snr,
            target_snrs=target_snrs,
            input_velocity_dispersions=input_velocity_dispersions,
            template_weight=template_weight,
            polynomial_degree=degree,
            polynomial_weights=polynomial_weights,
            multiplicative_polynomial_degree=ppxf_fit.mdegree,
            add_component=add_component,
            multiplicative_polynomial=ppxf_fit.mpoly if ppxf_fit.mdegree > 0 else 1.0,
            num_sample=num_samples,
            z_factor=z_factor,
        )

    @staticmethod
    def get_specific_signal_and_noise(spectra, mask, z_factor=1.0):
        """Get the mean SNR of the spectra.

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
        signal_slice = spectra.flux[mask]
        total_signal = np.sum(signal_slice)

        if spectra.covariance is not None:
            covariance_slice = spectra.covariance[mask][:, mask]
            total_noise = np.sqrt(np.sum(covariance_slice))
        else:
            total_noise = np.sqrt(np.sum(spectra.noise[mask] ** 2))

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
        polynomial_weights=[1.0],
        multiplicative_polynomial_degree=0,
        add_component=0.0,
        multiplicative_polynomial=1.0,
        num_sample=50,
        z_factor=1.0,
    ):
        """Check the bias in the velocity dispersion measurement as a function of SNR.

        :param data_object: data object
        :type data_object: squirrel.data.Spectra
        :param template_object: template object
        :type template_object: squirrel.template.Template
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
        :param polynomial_weights: weights of the polynomial
        :type polynomial_weights: numpy.ndarray
        :param add_component: additional component to add to the spectra
        :type add_component: numpy.ndarray
        :param multiply_component: multiplicative component for the spectra
        :type multiply_component: float
        :param num_samples: number of Monte Carlo noise realizations
        :type num_samples: int
        :param z_factor: multiplicative factor for wavelength (e.g., 1 + z), if the SNR
            is computed at a different frame
        :type z_factor: float
        :return: recovered values
        :rtype: tuple
        """
        if spectra_mask_for_snr is None:
            spectra_mask_for_snr = np.ones_like(spectra_data.wavelengths)

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

        data = deepcopy(spectra_data)
        data.noise = np.diag(data.covariance) ** 0.5
        data.covariance = None

        # for i, input_dispersion in enumerate(input_velocity_dispersions):
        for i in tqdm(range(len(input_velocity_dispersions)), desc="Input dispersion"):
            input_dispersion = input_velocity_dispersions[i]
            mock_flux = cls.make_convolved_spectra(
                template.flux[:, 0],
                template.wavelengths,
                input_dispersion,
                spectra_data.velocity_scale,
                int(spectra_data.velocity_scale / template.velocity_scale),
                spectra_data.wavelengths,
                template_weight,
                polynomial_degree,
                polynomial_weights,
                multiplicative_polynomial,
            )

            # mock_flux *= multiplicative_polynomial
            mock_flux += add_component

            # for j, target_snr in enumerate(target_snrs):
            for j in tqdm(range(len(target_snrs)), desc="Target SNR"):
                target_snr = target_snrs[j]
                dispersion_samples = np.zeros(num_sample)
                dispersion_uncertainty_samples = np.zeros(num_sample)
                velocity_samples = np.zeros(num_sample)
                velocity_uncertainty_samples = np.zeros(num_sample)
                snr_samples = np.zeros(num_sample)

                for k in range(num_sample):
                    data = deepcopy(spectra_data)
                    data.flux = deepcopy(mock_flux)

                    initial_specific_snr = cls.get_specific_snr(
                        spectra_data, spectra_mask_for_snr, z_factor=z_factor
                    )

                    noise_multiplier = initial_specific_snr / target_snr

                    if data.covariance is not None:
                        data.covariance *= noise_multiplier**2
                        noise = np.random.multivariate_normal(
                            data.flux * 0, data.covariance
                        )
                    elif data.noise is not None:
                        data.noise *= noise_multiplier
                        noise = np.random.normal(
                            data.flux * 0, data.noise, size=len(data.flux)
                        )
                    else:
                        raise ValueError("No noise or covariance provided.")

                    data.flux += noise

                    mock_ppxf_fit = Pipeline.run_ppxf(
                        data,
                        template,
                        moments=2,
                        degree=polynomial_degree,
                        mdegree=multiplicative_polynomial_degree,
                        start=[0, input_dispersion],
                        bounds=[
                            [-cls.VEL_LIM, cls.VEL_LIM],
                            [input_dispersion - 100, input_dispersion + 100],
                        ],
                        fixed=[1, 0],
                        quiet=True,
                        global_search={
                            "popsize": 20,
                            "mutation": (0.5, 1.0),
                            "disp": False,
                        },
                    )
                    if k == 0:
                        # plt.plot(data.flux)
                        # plt.plot(mock_flux)
                        # plt.plot(spectra_data.flux)
                        # plt.show()
                        mock_ppxf_fit.plot()
                        plt.title(f"input: {input_dispersion}, snr: {target_snr}")
                        plt.show()

                    # dispersion_samples.append(mock_ppxf_fit.sol[1])
                    # velocity_samples.append(mock_ppxf_fit.sol[0])
                    # snr_samples.append(cls.get_mean_snr(data, spectra_mask_for_snr))
                    dispersion_samples[k] = mock_ppxf_fit.sol[1]
                    dispersion_uncertainty_samples[k] = mock_ppxf_fit.error[1]
                    velocity_samples[k] = mock_ppxf_fit.sol[0]
                    velocity_uncertainty_samples[k] = mock_ppxf_fit.error[0]
                    snr_samples[k] = cls.get_specific_snr(
                        data, spectra_mask_for_snr, z_factor=z_factor
                    )

                # recovered_velocities[i, j] = np.sum(
                #     velocity_samples / velocity_uncertainty_samples**2
                # ) / np.sum(1 / velocity_uncertainty_samples**2)
                recovered_velocities[i, j] = np.median(velocity_samples)
                recovered_velocity_uncertainties[i, j] = (
                    np.sum(1 / velocity_uncertainty_samples**2) ** -0.5
                )
                # recovered_velocity_scatters[i, j] = np.std(velocity_samples)
                recovered_velocity_scatters[i, j] = 1.4826 * np.median(
                    np.abs(velocity_samples - recovered_velocities[i, j])
                )

                # recovered_dispersions[i, j] = np.sum(
                #     dispersion_samples / dispersion_uncertainty_samples**2
                # ) / np.sum(1 / dispersion_uncertainty_samples**2)
                recovered_dispersions[i, j] = np.median(dispersion_samples)
                recovered_dispersion_uncertainties[i, j] = (
                    np.sum(1 / dispersion_uncertainty_samples**2) ** -0.5
                )
                # recovered_dispersion_scatters[i, j] = np.std(dispersion_samples)
                recovered_dispersion_scatters[i, j] = 1.4826 * np.median(
                    np.abs(dispersion_samples - recovered_dispersions[i, j])
                )
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

    @classmethod
    def plot_bias_vs_snr(
        cls,
        recovered_values,
        input_velocity_dispersions,
        fig_width=10,
        bias_threshold=0.02,
        show_scatter=True,
        show_mean_uncertainty=False,
        **kwargs,
    ):
        """Plot the bias in the velocity dispersion measurement as a function of SNR.

        :param recovered_values: recovered values from check_bias_vs_snr
        :type recovered_values: tuple
        :param target_snrs: target SNRs
        :type target_snrs: numpy.ndarray
        :param input_velocity_dispersions: input velocity dispersions
        :type input_velocity_dispersions: numpy.ndarray
        :param fig_width: width of the figure
        :type fig_width: float
        :param bias_threshold: bias threshold line for plotting
        :type bias_threshold: float
        :param show_scatter: show scatter of the points
        :type show_scatter: bool
        :param plot_kwargs: keyword arguments for plotting
        :type plot_kwargs: dict
        :return: figure and axes
        :rtype: tuple
        """
        (
            recovered_snrs,
            recovered_dispersions,
            recovered_dispersion_uncertainties,
            recovered_dispersion_scatters,
            recovered_velocities,
            recovered_velocity_uncertainties,
            recovered_velocity_scatters,
        ) = recovered_values

        fig, axes = plt.subplots(
            len(input_velocity_dispersions),
            2,
            figsize=(fig_width, fig_width / 8.0 * len(input_velocity_dispersions)),
        )

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
        )

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
        )

        # remove gap between subplots
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
        **kwargs,
    ):
        """Plot the bias in the velocity dispersion measurement as a function of SNR.

        :param axes: axes to plot
        :type axes: numpy.ndarray
        :param input_values: input values
        :type input_values: numpy.ndarray
        :param recovered_snrs: recovered SNRs
        :type recovered_snrs: numpy.ndarray
        :param recovered_values: recovered values
        :type recovered_values: numpy.ndarray
        :param recovered_value_uncertainties: recovered value uncertainties
        :type recovered_value_uncertainties: numpy.ndarray
        :param recovered_value_scatters: recovered value scatters
        :type recovered_value_scatters: numpy.ndarray
        :param show_scatter: show scatter of the points
        :type show_scatter: bool
        :param show_mean_uncertainty: show mean uncertainty of the points
        :type show_mean_uncertainty: bool
        :param bias_threshold: bias threshold line for plotting
        :type bias_threshold: float
        :param x_label: x label
        :type x_label: str
        :param y_label: y label
        :type y_label: str
        :param kwargs: keyword arguments for plotting
        :type kwargs: dict
        """

        if len(input_values) == 1:
            axes = axes[np.newaxis, :]

        if "marker" not in kwargs:
            kwargs["marker"] = "o"
        if "ls" not in kwargs:
            kwargs["ls"] = ":"

        for i, input_value in enumerate(input_values):
            if show_scatter:
                axes[i].errorbar(
                    recovered_snrs[i],
                    recovered_values[i],
                    yerr=recovered_value_scatters[i],
                    markersize=5,
                    ecolor="grey",
                    capsize=3,
                    **kwargs,
                )
            if show_mean_uncertainty:
                axes[i].errorbar(
                    recovered_snrs[i],
                    recovered_values[i],
                    yerr=recovered_value_uncertainties[i],
                    capsize=8,
                    **kwargs,
                )

            axes[i].plot(
                recovered_snrs[i], recovered_values[i], mec="k", zorder=20, **kwargs
            )
            axes[i].axhline(input_value, color="grey", linestyle="--", alpha=0.6)

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
        template_wavelength,
        velocity_dispersion,
        velocity_scale,
        velocity_scale_ratio,
        data_wavelength,
        data_weight=1,
        polynomial_degree=0,
        polynomial_weights=[1.0],
        multiplicative_polynomial=1.0,
    ):
        """Make a convolved spectra.

        :param template_flux: template flux
        :type template_flux: numpy.ndarray
        :param template_wavelength: template wavelength
        :type template_wavelength: numpy.ndarray
        :param velocity_dispersion: velocity dispersion
        :type velocity_dispersion: float
        :param velocity_scale: velocity scale
        :type velocity_scale: float
        :param velocity_scale_ratio: velocity scale ratio
        :type velocity_scale_ratio: int
        :param data_npix: number of pixels in data
        :type data_npix: int
        :param data_wavelength: data wavelength
        :type data_wavelength: numpy.ndarray
        :param data_weight: multiplicative factor for the template flux, effective when
            polynomial_degree > 0 to set the relative amplitude of data and polynomial
        :type data_weight: float
        :param polynomial_degree: degree of the polynomial
        :type polynomial_degree: int
        :param polynomial_weights: weights of the polynomial
        :type polynomial_weights: numpy.ndarray
        :return: convolved spectra
        :rtype: numpy.ndarray
        """
        data_num_pix = len(data_wavelength)

        c = 299792.458  # speed of light in km/s
        lam_range = data_wavelength[[0, -1]] / np.exp(
            np.array([cls.VEL_LIM + 900, -cls.VEL_LIM - 900]) / c
        )  # Use eq.(5c) of Cappellari (2023)
        ok = (template_wavelength >= lam_range[0]) & (
            template_wavelength <= lam_range[1]
        )
        template_flux = template_flux[ok]
        template_wavelength = template_wavelength[ok]

        lam_temp_min = np.mean(template_wavelength[:velocity_scale_ratio])

        v_systemic = c * np.log(lam_temp_min / data_wavelength[0]) / velocity_scale
        template_npix = template_flux.shape[0]
        start = np.array([0, velocity_dispersion / velocity_scale])
        """
        nmin = max(self.templates.shape[0], self.npix)
        self.npad = 2 ** int(np.ceil(np.log2(nmin)))
        if templates_rfft is None:
            # Pre-compute FFT of real input of all templates
            self.templates_rfft = np.fft.rfft(self.templates, self.npad, axis=0)
        else:
            self.templates_rfft = templates_rfft
        """

        nmin = max(template_npix, data_num_pix)
        npad = 2 ** int(np.ceil(np.log2(nmin)))
        # npad = 2 ** int(np.ceil(np.log2(npix)))
        template_rfft = np.fft.rfft(template_flux, npad)

        lvd_rfft = losvd_rfft(
            start, 1, [2], template_rfft.shape[0], 1, v_systemic, 2, 0
        )

        template_convolved = np.fft.irfft(template_rfft * lvd_rfft[:, 0, 0], npad)[
            : data_num_pix * velocity_scale_ratio
        ]
        """Tmp = np.empty((nspec, self.npix)) for j, template_rfft in enumerate(

        self.templates_rfft.T ):  # loop over column templates     for k in
        range(nspec):         tt = np.fft.irfft(             template_rfft * lvd_rfft[:,
        self.component[j], k], self.npad         )         tmp[k, :] = rebin( tt[:
        self.npix * self.velscale_ratio], self.velscale_ratio         ) c[:nrows_spec,
        npoly + j] = tmp.ravel()
        """

        galaxy_model = rebin(
            template_convolved,
            velocity_scale_ratio,
        )

        x = np.linspace(-1, 1, data_num_pix)
        vand = legendre.legvander(x, polynomial_degree)

        return multiplicative_polynomial * data_weight * galaxy_model + np.dot(
            vand, polynomial_weights
        )
