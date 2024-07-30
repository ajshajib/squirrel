"""This module contains class and functions for diagnostics."""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy.polynomial import legendre
from ppxf.ppxf import losvd_rfft
from ppxf.ppxf import rebin

from .pipeline import Pipeline
from .template import Template


class Diagnostics(object):
    """This class contains functions to diagnose the performance of the pipeline."""

    @classmethod
    def check_bias_vs_snr_from_ppxf_fit(
        cls,
        ppxf_fit,
        spectra_data,
        spectra_mask_for_snr=None,
        target_snrs=np.arange(10, 31, 10),
        input_velocity_dispersions=[250],  # np.arange(100, 351, 50)
        num_samples=50,
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
        :param num_samples: number of Monte Carlo noise realizations
        :type num_samples: int
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
            num_sample=num_samples,
        )

    @staticmethod
    def get_mean_snr(spectra, mask):
        """Get the mean SNR of the spectra.

        :param spectra: spectra object
        :type spectra: squirrel.data.Spectra
        :param mask: mask
        :type mask: numpy.ndarray
        :return: mean SNR per wavelength unit
        :rtype: float
        """
        if "log_rebinned" in spectra.spectra_modifications:
            signal_slice = spectra.flux[mask] / spectra.wavelengths[mask]
            delta_lambda = np.log(spectra.wavelengths[1] / spectra.wavelengths[0])
        else:
            signal_slice = spectra.flux[mask]
            delta_lambda = np.mean(np.diff(spectra.wavelengths))

        if spectra.covariance is not None:
            covariance_slice = spectra.covariance[mask][:, mask]

            mean_snr = np.mean(
                np.dot(sp.linalg.sqrtm(np.linalg.inv(covariance_slice)), signal_slice),
            )
        elif spectra.noise is not None:
            noise_slice = spectra.noise[mask]
            mean_snr = np.mean(signal_slice / noise_slice)  # ** (1 / np.sum(mask))

        return mean_snr / delta_lambda

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
        num_sample=50,
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
        :param num_samples: number of Monte Carlo noise realizations
        :type num_sample: int
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

        recovered_dispersions = np.zeros(
            (len(input_velocity_dispersions), len(target_snrs))
        )
        recovered_dispersion_uncertainties = np.zeros(
            (len(input_velocity_dispersions), len(target_snrs))
        )

        recovered_snrs = np.zeros((len(input_velocity_dispersions), len(target_snrs)))

        data = deepcopy(spectra_data)

        for i, input_dispersion in enumerate(input_velocity_dispersions):
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
            )

            for j, target_snr in enumerate(target_snrs):
                dispersion_samples = np.zeros(num_sample)
                dispersion_uncertainty_samples = np.zeros(num_sample)
                velocity_samples = np.zeros(num_sample)
                velocity_uncertainty_samples = np.zeros(num_sample)
                snr_samples = np.zeros(num_sample)

                for k in range(num_sample):
                    data = deepcopy(spectra_data)
                    data.flux = deepcopy(mock_flux)

                    initial_mean_snr = cls.get_mean_snr(data, spectra_mask_for_snr)

                    noise_multiplier = initial_mean_snr / target_snr

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
                        start=[0, input_dispersion],
                        bounds=[
                            [-100, 100],
                            [input_dispersion - 100, input_dispersion + 100],
                        ],
                        mdegree=0,
                        quiet=True,
                        # global_search=True,  # {"popsize": 100, "mutation": (0.5, 1.9)},
                    )
                    # mock_ppxf_fit.plot()
                    # plt.show()

                    # dispersion_samples.append(mock_ppxf_fit.sol[1])
                    # velocity_samples.append(mock_ppxf_fit.sol[0])
                    # snr_samples.append(cls.get_mean_snr(data, spectra_mask_for_snr))
                    dispersion_samples[k] = mock_ppxf_fit.sol[1]
                    dispersion_uncertainty_samples[k] = mock_ppxf_fit.error[1]
                    velocity_samples[k] = mock_ppxf_fit.sol[0]
                    velocity_uncertainty_samples[k] = mock_ppxf_fit.error[0]
                    snr_samples[k] = cls.get_mean_snr(data, spectra_mask_for_snr)

                recovered_velocities[i, j] = np.sum(
                    velocity_samples / velocity_uncertainty_samples**2
                ) / np.sum(1 / velocity_uncertainty_samples**2)
                recovered_velocity_uncertainties[i, j] = (
                    np.sum(1 / velocity_uncertainty_samples**2) ** -0.5
                )

                recovered_dispersions[i, j] = np.sum(
                    dispersion_samples / dispersion_uncertainty_samples**2
                ) / np.sum(1 / dispersion_uncertainty_samples**2)
                recovered_dispersion_uncertainties[i, j] = (
                    np.sum(1 / dispersion_uncertainty_samples**2) ** -0.5
                )
                recovered_snrs[i, j] = np.mean(snr_samples)

        return (
            recovered_snrs,
            recovered_dispersions,
            recovered_dispersion_uncertainties,
            recovered_velocities,
            recovered_velocity_uncertainties,
        )

    @staticmethod
    def plot_bias_vs_snr(
        recovered_values,
        input_velocity_dispersions,
        fig_width=10,
        bias_threshold=0.02,
        **kwargs
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
        :param plot_kwargs: keyword arguments for plotting
        :type plot_kwargs: dict
        :return: figure and axes
        :rtype: tuple
        """
        (
            recovered_snrs,
            recovered_dispersions,
            recovered_dispersion_uncertainties,
            recovered_velocities,
            recovered_velocity_uncertainties,
        ) = recovered_values

        fig, axes = plt.subplots(
            len(input_velocity_dispersions),
            2,
            figsize=(fig_width, fig_width / 8.0 * len(input_velocity_dispersions)),
        )
        if len(input_velocity_dispersions) == 1:
            axes = axes[np.newaxis, :]

        if "marker" not in kwargs:
            kwargs["marker"] = "o"
        if "ls" not in kwargs:
            kwargs["ls"] = ":"

        for i, input_dispersion in enumerate(input_velocity_dispersions):
            # axes[i, 0].errorbar(
            #     target_snrs,
            #     recovered_dispersions[i],
            #     yerr=recovered_velocity_scatters[i],
            #     markersize=5,
            #     ecolor="grey",
            #     capsize=5,
            #     **kwargs,
            # )
            axes[i, 0].errorbar(
                recovered_snrs[i],
                recovered_dispersions[i],
                yerr=recovered_velocity_uncertainties[i],
                capsize=8,
                **kwargs,
            )

            axes[i, 0].axhline(input_dispersion, color="grey", linestyle="--")
            axes[i, 0].axhspan(
                input_dispersion * (1 - bias_threshold),
                input_dispersion * (1 + bias_threshold),
                color="grey",
                alpha=0.5,
                zorder=-10,
            )
            axes[i, 0].set_ylabel(r"$\sigma$ (km s$^{-1}$)")

            # axes[i, 1].errorbar(
            #     target_snrs,
            #     recovered_velocities[i],
            #     yerr=recovered_velocity_scatters[i],
            #     markersize=5,
            #     ecolor="grey",
            #     capsize=5,
            #     **kwargs,
            # )
            axes[i, 1].errorbar(
                recovered_snrs[i],
                recovered_velocities[i],
                yerr=recovered_velocity_uncertainties[i],
                capsize=8,
                **kwargs,
            )
            axes[i, 1].axhline(0, color="k", linestyle="--")
            axes[i, 1].set_ylabel(r"$\Delta v$ (km s$^{-1}$)")

        axes[-1, 0].set_xlabel(r"$S/N$")
        axes[-1, 1].set_xlabel(r"$S/N$")
        # remove gap between subplots
        plt.subplots_adjust(hspace=0.0)

        return fig, axes

    @staticmethod
    def make_convolved_spectra(
        template_flux,
        template_wavelength,
        velocity_dispersion,
        velocity_scale,
        velocity_scale_ratio,
        data_wavelength,
        data_weight=1,
        polynomial_degree=0,
        polynomial_weights=[1.0],
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
            np.array([2900, -2900]) / c
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

        galaxy_model = rebin(
            template_convolved,
            velocity_scale_ratio,
        )

        x = np.linspace(-1, 1, data_num_pix)
        vand = legendre.legvander(x, polynomial_degree)

        return data_weight * galaxy_model + np.dot(vand, polynomial_weights)
