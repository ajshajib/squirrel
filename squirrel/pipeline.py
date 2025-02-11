"""This module contains the class to wrap the pPXF package for kinematic analysis."""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.special import ndtr
from ppxf import ppxf_util
from ppxf.ppxf import ppxf
from ppxf import sps_util
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from vorbin.voronoi_2d_binning import _compute_useful_bin_quantities
from vorbin.voronoi_2d_binning import _sn_func
from tqdm import tqdm

from .data import VoronoiBinnedSpectra
from .template import Template
from .util import is_positive_definite
from .util import get_nearest_positive_definite_matrix


class Pipeline(object):
    """A class to wrap the pPXF package for kinematic analysis."""

    _speed_of_light = 299792.458  # speed of light in km/s

    @staticmethod
    def log_rebin(
        spectra,
        velocity_scale=None,
        num_samples_for_covariance=None,
        take_covariance=True,
    ):
        """Rebin the data to log scale.

        :param data: data to rebin
        :type data: `Data` class
        :param velocity_scale: velocity scale for the rebinning
        :type velocity_scale: float
        :param num_samples_for_covariance: number of samples for the covariance estimation
        :type num_samples_for_covariance: int
        :param take_covariance: take the covariance into account
        :type take_covariance: bool
        """
        if "log_rebinned" in spectra.spectra_modifications:
            raise ValueError("Data has already been log rebinned.")

        wavelength_range = spectra.wavelengths[[0, -1]]
        rebinned_spectra, log_rebinned_wavelength, velocity_scale = ppxf_util.log_rebin(
            wavelength_range, spectra.flux, velscale=velocity_scale
        )

        # estimate covariance matrix of rebinned spectra through sampling from noise realizations
        if num_samples_for_covariance is None:
            # if number of samples not provided, picking the data size following this
            # paper: https://arxiv.org/abs/1004.3484
            num_samples_for_covariance = spectra.flux.shape[0]

        flux_shape = spectra.flux.shape
        flux_flattenned = np.atleast_2d(spectra.flux.reshape(flux_shape[0], -1))
        noise_flattenned = np.atleast_2d(spectra.noise.reshape(flux_shape[0], -1))
        if flux_flattenned.shape[0] == 1:
            flux_flattenned = flux_flattenned.T
            noise_flattenned = noise_flattenned.T

        if take_covariance:
            covariance = np.atleast_3d(
                np.zeros(
                    (
                        rebinned_spectra.shape[0],
                        rebinned_spectra.shape[0],
                        *flux_flattenned.shape[1:],
                    )
                )
            )
            noise = None
        else:
            covariance = None
            noise = np.atleast_2d(
                np.zeros((rebinned_spectra.shape[0], *flux_flattenned.shape[1:]))
            )

        for i in tqdm(range(flux_flattenned.shape[1])):
            flux_realizations = np.random.normal(
                flux_flattenned[:, i],
                noise_flattenned[:, i],
                (num_samples_for_covariance, len(flux_flattenned[:, i])),
            ).T
            rebinned_realizations, _, _ = ppxf_util.log_rebin(
                wavelength_range, flux_realizations, velscale=velocity_scale
            )

            if take_covariance:
                covariance[:, :, i] = np.cov(rebinned_realizations)

                # set non-adjacent covariance to zero to avoid numerical issues from noise
                for j in range(covariance.shape[0]):
                    for k in range(covariance.shape[1]):
                        if np.abs(j - k) > 1:
                            covariance[j, k, i] = 0
            else:
                noise[:, i] = np.std(rebinned_realizations, axis=1)

        if take_covariance:
            # reverse flattened array
            covariance = covariance.reshape(
                rebinned_spectra.shape[0], rebinned_spectra.shape[0], *flux_shape[1:]
            )
        else:
            noise = noise.reshape(rebinned_spectra.shape[0], *flux_shape[1:])

        spectra.flux = rebinned_spectra
        spectra.noise = noise
        spectra.covariance = covariance
        spectra.wavelengths = np.exp(log_rebinned_wavelength)
        spectra.velocity_scale = velocity_scale
        spectra.spectra_modifications += ["log_rebinned"]

    @staticmethod
    def get_voronoi_binning_map(
        datacube,
        signal_image_per_wavelength_unit,
        noise_image,
        target_snr,
        max_radius,
        min_snr_per_spaxel=1.0,
        plot=False,
        quiet=True,
    ):
        """Get the Voronoi binning map.

        :param datacube: datacube to bin
        :type datacube: `DataCube` class
        :param signal_image_per_wavelength_unit: signal image per wavelength unit
        :type signal_image_per_wavelength_unit: np.ndarray
        :param noise_image: noise image
        :type noise_image: np.ndarray
        :param target_snr: target S/N per wavelength unit for each bin
        :type target_snr: float
        :param max_radius: maximum radius for binning, in the unit of in `datacube.x_coordinates`
        :type max_radius: float
        :param min_snr_per_spaxel: minimum S/N per spaxel to include in the binning
        :type min_snr_per_spaxel: float
        :param plot: plot the results
        :type plot: bool
        :param quiet: suppress the output
        """

        radius = np.sqrt(datacube.x_coordinates**2 + datacube.y_coordinates**2)

        snr_image_per_wavelength_unit = signal_image_per_wavelength_unit / noise_image
        snr_mask = (snr_image_per_wavelength_unit > min_snr_per_spaxel) & (
            radius < max_radius
        )

        x_pixels = np.arange(datacube.flux.shape[2])
        y_pixels = np.arange(datacube.flux.shape[1])
        xx_pixels, yy_pixels = np.meshgrid(x_pixels, y_pixels)

        xx_pixels_masked = xx_pixels[snr_mask]
        yy_pixels_masked = yy_pixels[snr_mask]

        xx_coordinates_masked = datacube.x_coordinates[snr_mask]
        yy_coordinates_masked = datacube.y_coordinates[snr_mask]

        signal_image_per_wavelength_unit_masked = signal_image_per_wavelength_unit[
            snr_mask
        ]
        noise_image_masked = noise_image[snr_mask]

        (
            num_bins,
            x_node,
            y_node,
            bin_center_x,
            bin_center_y,
            snr,
            n_pixels,
            scale,
        ) = voronoi_2d_binning(
            xx_coordinates_masked,
            yy_coordinates_masked,
            signal_image_per_wavelength_unit_masked,
            noise_image_masked,
            target_snr,
            plot=plot,
            quiet=quiet,
        )

        classes, x_bar, y_bar, snr, area = _compute_useful_bin_quantities(
            xx_coordinates_masked,
            yy_coordinates_masked,
            signal_image_per_wavelength_unit_masked,
            noise_image_masked,
            x_node,
            y_node,
            scale,
            sn_func=_sn_func,
        )

        if plot:
            plt.tight_layout()

        return (
            num_bins,
            xx_pixels_masked,
            yy_pixels_masked,
            bin_center_x,
            bin_center_y,
            snr,
            area,
        )

    @staticmethod
    def get_voronoi_binned_spectra(datacube, bin_mapping_output):
        """Perform the Voronoi binning.

        :param datacube: datacube to bin
        :type datacube: `DataCube` class
        :param bin_mapping_output: outputs from `get_voronoi_binning_map()`
        :type bin_mapping_output: tuple
        :return: Voronoi binned spectra
        :rtype: `VoronoiBinnedSpectra` class
        """
        (
            num_bins,
            xx_pixels_masked,
            yy_pixels_masked,
            bin_center_x,
            bin_center_y,
            snr,
            area,
        ) = bin_mapping_output

        voronoi_binned_flux = np.zeros(
            (datacube.flux.shape[0], int(np.max(num_bins)) + 1)
        )
        if datacube.noise is not None:
            voronoi_binned_noise = np.zeros_like(voronoi_binned_flux)
        else:
            voronoi_binned_noise = None
        if datacube.covariance is not None:
            voronoi_binned_covariance = np.zeros(
                (
                    voronoi_binned_flux.shape[0],
                    voronoi_binned_flux.shape[0],
                    int(np.max(num_bins)) + 1,
                )
            )
        else:
            voronoi_binned_covariance = None

        # for i in range(voronoi_bins.shape[0]):
        for x, y, n_bin in zip(xx_pixels_masked, yy_pixels_masked, num_bins):
            voronoi_binned_flux[:, n_bin] += datacube.flux[:, y, x]
            if datacube.noise is not None:
                voronoi_binned_noise[:, n_bin] += datacube.noise[:, y, x] ** 2
            if datacube.covariance is not None:
                voronoi_binned_covariance[:, :, n_bin] += datacube.covariance[
                    :, :, y, x
                ]

        if datacube.noise is not None:
            voronoi_binned_noise = np.sqrt(voronoi_binned_noise)

        voronoi_binned_spectra = VoronoiBinnedSpectra(
            wavelengths=datacube.wavelengths,
            flux=voronoi_binned_flux,
            wavelength_unit=datacube.wavelength_unit,
            fwhm=datacube.fwhm,
            z_lens=datacube.z_lens,
            z_source=datacube.z_source,
            x_coordinates=datacube.x_coordinates,
            y_coordinates=datacube.y_coordinates,
            num_bins=num_bins,
            x_pixels_of_bins=xx_pixels_masked,
            y_pixels_of_bins=yy_pixels_masked,
            flux_unit=datacube.flux_unit,
            noise=voronoi_binned_noise,
            covariance=voronoi_binned_covariance,
            bin_center_x=bin_center_x,
            bin_center_y=bin_center_y,
            area=area,
            snr=snr,
        )

        voronoi_binned_spectra.spectra_modifications = deepcopy(
            datacube.spectra_modifications
        )
        voronoi_binned_spectra.wavelengths_frame = deepcopy(datacube.wavelengths_frame)
        voronoi_binned_spectra.velocity_scale = deepcopy(datacube.velocity_scale)

        return voronoi_binned_spectra

    @staticmethod
    def create_kinematic_map_from_bins(bin_mapping, kinematic_values):
        """Create a kinematic map from the binned spectra and the kinematic values.

        :param bin_mapping: a 2D array showing bin numbers for each pixel
        :type bin_mapping: np.ndarray
        :param kinematic_values: kinematic values
        :type kinematic_values: list of float
        :return: kinematic map
        :rtype: np.ndarray
        """
        kinematic_map = np.zeros_like(bin_mapping)

        for i in range(kinematic_map.shape[0]):
            for j in range(kinematic_map.shape[1]):
                if bin_mapping[i, j] == -1:
                    continue

                kinematic_map[i, j] = kinematic_values[int(bin_mapping[i, j])]

        return kinematic_map

    @staticmethod
    def get_template_from_library(
        library_path,
        spectra,
        velocity_scale_ratio,
        wavelength_factor=1.0,
        wavelength_range_extend_factor=1.05,
        **kwargs,
    ):
        """Get the template object created for a stellar template library. The `library_path` should point to a `numpy.savez()` file containing the following arrays for a given SPS models library, like FSPS, Miles, GALEXEV, BPASS. This file will be sent to `ppxf.sps_util.sps_lib()`. See the documentation of that function for the format of the file.
        The EMILES, FSPS, GALEXEV libraries are available at https://github.com/micappe/ppxf_data.

        :param library_path: path to the library
        :type library_path: str
        :param spectra: log rebinned spectra, which the templates will be used for
        :type spectra: `Spectra` or a child class
        :param velocity_scale_ratio: velocity scale ratio for the template
        :type velocity_scale_ratio: float
        :param wavelength_factor: factor to multiply the wavelength range to get the templates for, used for de-redshifting, if necessary
        :type wavelength_factor: float
        :param wavelength_range_extend_factor: factor to extend the wavelength range
        :type wavelength_range_extend_factor: float
        :param kwargs: additional arguments for `ppxf.sps_util.sps_lib()` function
        :type kwargs: dict
        :return: template
        :rtype: `Template` class
        """
        assert spectra.wavelength_unit == "AA", "Wavelength unit must be in Angstrom."
        assert (
            "log_rebinned" in spectra.spectra_modifications
        ), "Data must be log rebinned."

        wavelength_range_templates = (
            spectra.wavelengths[0] / wavelength_range_extend_factor * wavelength_factor,
            spectra.wavelengths[-1]
            * wavelength_range_extend_factor
            * wavelength_factor,
        )

        # template library will be sampled at data resolution times the velscale_ratio in the given wavelength range
        sps = sps_util.sps_lib(
            library_path,
            spectra.velocity_scale / velocity_scale_ratio,
            spectra.fwhm,
            lam_range=wavelength_range_templates,
            norm_range=[
                spectra.wavelengths[0] * wavelength_factor,
                spectra.wavelengths[-1] * wavelength_factor,
            ],
            **kwargs,
        )

        template_fluxes = sps.templates.reshape(sps.templates.shape[0], -1)
        templates_wavelengths = sps.lam_temp / wavelength_factor

        template = Template(
            templates_wavelengths,
            template_fluxes,
            wavelength_unit="AA",
            fwhm=spectra.fwhm,
        )
        template.velocity_scale = spectra.velocity_scale / velocity_scale_ratio

        return template

    @staticmethod
    def get_emission_line_template(
        spectra,
        template_wavelengths,
        wavelength_factor=1.0,
        wavelength_range_extend_factor=1.05,
        **kwargs,
    ):
        """Get the emission line template.

        :param spectra: log rebinned spectra
        :type spectra: `Spectra` or a child class
        :param template_wavelengths: wavelengths of the template in Angstrom
        :type template_wavelengths: np.ndarray
        :param wavelength_factor: factor to multiply the wavelength range to get the templates for, used for de-redshifting, if necessary
        :type wavelength_factor: float
        :param wavelength_range_extend_factor: factor to extend the wavelength range
        :type wavelength_range_extend_factor: float
        :param kwargs: additional arguments for `ppxf_util.emission_lines`
        :type kwargs: dict
        :return: emission line template
        :rtype: `Template` class
        """
        wavelength_range_templates = (
            spectra.wavelengths[0] / wavelength_range_extend_factor * wavelength_factor,
            spectra.wavelengths[-1]
            * wavelength_range_extend_factor
            * wavelength_factor,
        )

        gas_templates, line_names, line_wavelengths = ppxf_util.emission_lines(
            np.log(template_wavelengths * wavelength_factor),
            wavelength_range_templates,
            spectra.fwhm,
            **kwargs,
        )

        template = Template(
            template_wavelengths,
            gas_templates,
            wavelength_unit="AA",
            fwhm=spectra.fwhm,
        )

        return template, line_names, line_wavelengths

    @staticmethod
    def join_templates(
        kinematic_template,
        kinematic_template_2=None,
        emission_line_template=None,
        emission_line_groups=None,
    ):
        flux = kinematic_template.flux
        component_indices = np.zeros(kinematic_template.flux.shape[1], dtype=int)

        if kinematic_template_2 is not None:
            if len(flux.shape) > len(kinematic_template_2.flux.shape):
                kinematic_template_2.flux = np.expand_dims(
                    kinematic_template_2.flux, axis=1
                )
            elif len(flux.shape) < len(kinematic_template_2.flux.shape):
                flux = np.expand_dims(flux, axis=1)
            flux = np.append(flux, kinematic_template_2.flux, axis=1)
            component_indices = np.append(
                component_indices,
                np.ones(kinematic_template_2.flux.shape[1], dtype=int),
            )

        if emission_line_template is not None:
            flux = np.append(flux, emission_line_template.flux, axis=1)
            if kinematic_template_2 is not None:
                component_indices = np.append(
                    component_indices, emission_line_groups + 2
                )
                emission_line_indices = component_indices > 1.0
            else:
                component_indices = np.append(
                    component_indices, emission_line_groups + 1
                )
                emission_line_indices = component_indices > 0.0
        else:
            emission_line_indices = np.zeros_like(component_indices, dtype=bool)

        template = Template(
            kinematic_template.wavelengths,
            flux,
            wavelength_unit=kinematic_template.wavelength_unit,
            fwhm=kinematic_template.fwhm,
        )
        template.velocity_scale = kinematic_template.velocity_scale

        return template, component_indices, emission_line_indices

    @staticmethod
    def make_template_from_array(
        fluxes,
        wavelengths,
        fwhm_template,
        spectra,
        velocity_scale_ratio,
        wavelength_factor=1.0,
        wavelength_range_extend_factor=1.05,
    ):
        """Get the template object from the given fluxes and wavelengths.

        :param fluxes: fluxes of the templates, dimensions must be (n_wavelengths, n_templates)
        :type fluxes: np.ndarray
        :param wavelengths: wavelengths of the templates in Angstrom
        :type wavelengths: np.ndarray
        :param spectra: log rebinned spectra, which the templates will be used for
        :type spectra: `Spectra` or a child class
        :param velocity_scale_ratio: velocity scale ratio for the template
        :type velocity_scale_ratio: float
        :param wavelength_factor: factor to multiply the wavelength range to get the templates for, used for de-redshifting, if necessary
        :type wavelength_factor: float
        :param wavelength_range_extend_factor: factor to extend the wavelength range
        :type wavelength_range_extend_factor: float
        :return: template
        :rtype: `Template` class
        """
        assert spectra.wavelength_unit == "AA", "Wavelength unit must be in Angstrom."
        assert (
            "log_rebinned" in spectra.spectra_modifications
        ), "Data must be log rebinned."

        wavelength_min = (
            spectra.wavelengths[0] / wavelength_range_extend_factor * wavelength_factor
        )
        wavelength_max = (
            spectra.wavelengths[-1] * wavelength_range_extend_factor * wavelength_factor
        )

        wavelength_diff = np.mean(np.diff(wavelengths))

        fluxes = fluxes[
            (wavelengths > wavelength_min - wavelength_diff)
            & (wavelengths < wavelength_max + wavelength_diff),
            :,
        ]
        wavelengths = wavelengths[
            (wavelengths > wavelength_min - wavelength_diff)
            & (wavelengths < wavelength_max + wavelength_diff)
        ]

        if fwhm_template < spectra.fwhm:
            sigma_diff = (
                np.sqrt(spectra.fwhm**2 - fwhm_template**2) / 2.355 / wavelength_diff
            )
            convolved_fluxes = ndimage.gaussian_filter1d(fluxes, sigma_diff, axis=0)

        rebinned_fluxes, log_wavelengths, velocity_scale_template = ppxf_util.log_rebin(
            wavelengths,
            convolved_fluxes,
            velscale=spectra.velocity_scale / velocity_scale_ratio,
        )

        rebinned_fluxes /= np.nanmean(rebinned_fluxes, axis=0)

        templates_wavelengths = np.exp(log_wavelengths) / wavelength_factor

        template = Template(
            templates_wavelengths,
            rebinned_fluxes,
            wavelength_unit="AA",
            fwhm=spectra.fwhm,
        )
        template.velocity_scale = velocity_scale_template

        return template

    @staticmethod
    def run_ppxf(
        data,
        template,
        start,
        background_template=None,
        spectra_indices=None,
        quiet=True,
        plot=False,
        **kwargs_ppxf,
    ):
        """Perform the kinematic analysis using pPXF.

        :param data: data to analyze
        :type data: `Data` class
        :param template_library: library of templates
        :type template_library: `TemplateLibrary` class
        :param start: initial guess for the velocity and dispersion for each kinematic component, , check documentation of `ppxf.ppxf()`
        :type start: list
        :param velocity_scale_ratio: ratio of the velocity scale to the velocity dispersion
        :type velocity_scale_ratio: float
        :param background_template: background spectra to fit
        :type background_template: `Data` class
        :param spectra_indices: indices of the spectra to fit, used for datacubes or binned spectra
        :type spectra_indices: list of int or int
        :param quiet: suppress the output
        :type quiet: bool
        :param kwargs_ppxf: additional options for `ppxf`, check documentation of `ppxf.ppxf()`
        :type kwargs_ppxf: dict
        :return: pPXF fit
        :rtype: `ppxf` class
        """
        noise = None

        if spectra_indices is not None:
            if isinstance(spectra_indices, int) and data.flux.ndim == 2:
                assert (
                    data.flux.ndim == 2
                ), f"Spectra indices must be an integer for spectra with {data.spectra.ndim} dimensions."
                flux = data.flux[:, spectra_indices]
                if data.covariance is not None:
                    noise = data.covariance[:, :, spectra_indices]
                elif data.noise is not None:
                    noise = data.noise[:, spectra_indices]
            elif (
                isinstance(spectra_indices, list)
                and len(spectra_indices) == data.flux.ndim - 1
                and data.flux.ndim == 3
            ):
                flux = data.flux[:, spectra_indices[0], spectra_indices[1]]
                if data.covariance is not None:
                    noise = data.covariance[
                        :, :, spectra_indices[0], spectra_indices[1]
                    ]
                elif data.noise is not None:
                    noise = data.noise[:, spectra_indices[0], spectra_indices[1]]
            else:
                if data.flux.ndim == 2:
                    raise ValueError(
                        f"Spectra indices must be an integer for spectra with {data.flux.ndim} dimensions."
                    )
                else:
                    raise ValueError(
                        f"Spectra indices must be a list of {data.flux.ndim - 1} integers for spectra with {data.flux.ndim} dimensions."
                    )
        else:
            flux = data.flux
            if data.covariance is not None:
                noise = data.covariance
            elif data.noise is not None:
                noise = data.noise

        if noise is None:
            noise = 0.1 * np.ones_like(flux)

        if len(noise.shape) == 2 and not is_positive_definite(noise):
            noise = get_nearest_positive_definite_matrix(noise)

        original_noise = deepcopy(noise)
        ppxf_fit = ppxf(
            templates=template.flux,
            galaxy=flux,
            noise=deepcopy(
                noise
            ),  # sending deepcopy just in case, as pPXF may manipulate the noise array/matrix
            velscale=data.velocity_scale,
            start=start,
            plot=plot,
            lam=data.wavelengths,
            lam_temp=template.wavelengths,
            sky=background_template.flux if background_template else None,
            quiet=quiet,
            velscale_ratio=int(data.velocity_scale / template.velocity_scale),
            **kwargs_ppxf,
        )

        ppxf_fit.original_noise = original_noise
        return ppxf_fit

    @classmethod
    def run_ppxf_on_binned_spectra(
        cls,
        binned_spectra,
        template,
        start,
        background_template=None,
        **kwargs_ppxf,
    ):
        """Perform the kinematic analysis using pPXF on binned spectra.

        :param binned_spectra: binned spectra to analyze
        :type binned_spectra: `VoronoiBinnedSpectra` class
        :param template_library: library of templates
        :type template_library: `TemplateLibrary` class
        :param velocity_dispersion_guess: initial guess for the velocity dispersion
        :type velocity_dispersion_guess: float
        :param degree: degree of the additive polynomial
        :type degree: int
        :param velocity_scale_ratio: ratio of the velocity scale to the velocity dispersion
        :type velocity_scale_ratio: float
        :param background_template: background template to fit
        :type background_template: `Template` class
        :param spectra_indices: indices of the spectra to fit
        :type spectra_indices: list of int or int
        :param kwargs: additional arguments for `ppxf`
        :type kwargs: dict
        :return: velocity dispersions, velocity dispersion uncertainties, mean velocities, mean velocity uncertainties
        :rtype: tuple of np.ndarray
        """
        num_spectra = binned_spectra.flux.shape[1]

        velocity_dispersions = []
        velocity_dispersion_uncertainties = []
        mean_velocities = []
        mean_velocity_uncertainties = []

        for i in range(num_spectra):
            ppxf_fit = cls.run_ppxf(
                binned_spectra,
                template,
                start=start,
                background_template=background_template,
                spectra_indices=i,
                **kwargs_ppxf,
            )

            velocity_dispersions.append(ppxf_fit.sol[1])
            velocity_dispersion_uncertainties.append(ppxf_fit.error[1])
            mean_velocities.append(ppxf_fit.sol[0])
            mean_velocity_uncertainties.append(ppxf_fit.error[0])

        return (
            np.array(velocity_dispersions),
            np.array(velocity_dispersion_uncertainties),
            np.array(mean_velocities),
            np.array(mean_velocity_uncertainties),
        )

    @staticmethod
    def get_terms_in_bic(ppxf_fit, num_fixed_parameters=0, weight_threshold=0.01):
        """
        Get the k, n, and log_L terms that is needed to compute the BIC.

        :param ppxf_fit: ppxf fit object
        :type ppxf_fit: ppxf.ppxf
        :param num_fixed_parameters: number of fixed parameters in `fixed` given to ppxf
        :type num_fixed_parameters: int
        :return: k, n, log_L
        :rtype: tuple of int, int, float
        """
        n = len(ppxf_fit.goodpixels)

        if weight_threshold is not None:
            num_templates = np.sum(
                ppxf_fit.weights > weight_threshold * ppxf_fit.weights.sum()
            )
        else:
            num_templates = len(ppxf_fit.weights)

        k_linear = num_templates + ppxf_fit.degree + 1
        if ppxf_fit.sky is not None:
            k_linear += ppxf_fit.sky.shape[1]

        # check if  ppxf_fit.sol[0] is float
        if isinstance(ppxf_fit.sol[0], float):
            k_non_linear = len(ppxf_fit.sol) + ppxf_fit.mdegree
        else:
            k_non_linear = np.sum([len(a) for a in ppxf_fit.sol]) + ppxf_fit.mdegree

        k = k_linear + k_non_linear - num_fixed_parameters

        # likelihood computation based on ppxf numbers
        # dof = ppxf_fit.dof
        # chi2 = ppxf_fit.chi2
        # log_likelihood = -0.5 * ppxf_fit.chi2 * ppxf_fit.dof

        # likelihood computation based on residuals
        # dof = len(ppxf_fit.galaxy)  # - len(ppxf_fit.weights)
        residuals = ppxf_fit.galaxy - ppxf_fit.bestfit

        mask = np.zeros_like(residuals)
        mask[ppxf_fit.goodpixels] = 1
        residuals = residuals[mask == 1]
        covariance = np.copy(ppxf_fit.original_noise)
        covariance = covariance[mask == 1][:, mask == 1]

        chi2 = np.dot(residuals, np.dot(np.linalg.inv(covariance), residuals))
        log_likelihood = -0.5 * chi2

        return k, n, log_likelihood

    @classmethod
    def get_bic(cls, ppxf_fit, num_fixed_parameters=0, weight_threshold=0.01):
        """
        :param ppxf_fit: ppxf fit object
        :type ppxf_fit: ppxf.ppxf
        :param num_fixed_parameters: number of fixed parameters in `fixed` given to ppxf
        :type num_fixed_parameters: int
        :param weight_threshold: threshold for the weights. Default is 1% (0.01).
        :type weight_threshold: float
        :return: BIC
        :rtype: float
        """
        k, n, log_likelihood = cls.get_terms_in_bic(
            ppxf_fit,
            num_fixed_parameters=num_fixed_parameters,
            weight_threshold=weight_threshold,
        )
        bic = k * np.log(n) - 2 * log_likelihood

        return bic

    @classmethod
    def get_bic_from_sample(
        cls, ppxf_fits, num_fixed_parameters=0, weight_threshold=0.01
    ):
        """
        :param ppxf_fits: ppxf fit objects
        :type ppxf_fits: list of ppxf.ppxf
        :param num_fixed_parameters: number of fixed parameters in `fixed` given to ppxf
        :type num_fixed_parameters: int
        :param weight_threshold: threshold for the weights. Default is 1% (0.01).
        :type weight_threshold: float
        :return: BIC
        :rtype: float
        """
        k_total = 0
        n_total = 0
        log_likelihood_total = 0

        for ppxf_fit in ppxf_fits:
            k, n, log_likelihood = cls.get_terms_in_bic(
                ppxf_fit,
                num_fixed_parameters=num_fixed_parameters,
                weight_threshold=weight_threshold,
            )
            k_total += k
            n_total += n
            log_likelihood_total += log_likelihood

        bic = k_total * np.log(n_total) - 2 * log_likelihood_total

        return bic

    @classmethod
    def get_relative_bic_weights_for_sample(
        cls,
        ppxf_fits_list,
        num_fixed_parameters=0,
        num_bootstrap_samples=1000,
        weight_threshold=0.01,
    ):
        """
        Calculate the relative BIC weights for a given sample of pPXF fits.

        :param ppxf_fits_list: The sample of pPXF fits.
        :type ppxf_fits_list: np.ndarray
        :param num_fixed_parameters: The number of fixed parameters in the model.
        :type num_fixed_parameters: int
        :param num_bootstrap_samples: The number of bootstrap samples to use.
        :type num_bootstrap_samples: int
        :param weight_threshold: The threshold for the relative BIC weights. Default is 1% (0.01).
        :type weight_threshold: float

        """
        bics = np.zeros(len(ppxf_fits_list))
        weights = np.zeros_like(bics)

        for i, ppxf_fits in enumerate(ppxf_fits_list):
            bics[i] = cls.get_bic_from_sample(
                ppxf_fits,
                num_fixed_parameters=num_fixed_parameters,
                weight_threshold=weight_threshold,
            )

        delta_bics = bics - np.min(bics)

        # do bootstrap sampling
        bics_samples = np.zeros((num_bootstrap_samples, len(ppxf_fits_list)))
        for i in range(num_bootstrap_samples):
            indices = np.random.randint(
                0, len(ppxf_fits_list[0]), len(ppxf_fits_list[0])
            )
            ppxf_fits_list_bootstrapped = ppxf_fits_list[:, indices]

            for j, ppxf_fits in enumerate(ppxf_fits_list):
                bics_samples[i, j] = Pipeline.get_bic_from_sample(
                    ppxf_fits_list_bootstrapped[j],
                    num_fixed_parameters=num_fixed_parameters,
                    weight_threshold=weight_threshold,
                )

        bics_uncertainty = np.std(bics_samples, axis=0)

        for i in range(len(bics)):
            weights[i] = cls.calculate_weights_from_bic(
                delta_bics[i], bics_uncertainty[i]
            )

        return weights

    @classmethod
    def combine_measurements_from_templates(
        cls,
        values,
        uncertanties,
        ppxf_fits_list,
        apply_bic_weighting=True,
        num_fixed_parameters=0,
        num_bootstrap_samples=1000,
        weight_threshold=0.01,
        verbose=False,
    ):
        """
        Combine measurements using the relative BIC weights.

        :param values: The values to combine, can be a sample average values for different templates, or array of values (in which case should match the ppxf_fits_list shape).
        :type values: np.ndarray
        :param uncertanties: The uncertainties in the values.
        :type uncertanties: np.ndarray
        :param ppxf_fits_list: The list of pPXF fits.
        :type ppxf_fits_list: np.ndarray
        :param apply_bic_weighting: Whether to apply BIC weighting.
        :type apply_bic_weighting: bool
        :param num_fixed_parameters: The number of fixed parameters in the model.
        :type num_fixed_parameters: int
        :param num_bootstrap_samples: The number of bootstrap samples to use.
        :type num_bootstrap_samples: int
        :param weight_threshold: The threshold for the relative BIC weights. Default is 1% (0.01).
        :type weight_threshold: float
        :param verbose: Whether to print the results.
        :type verbose: bool
        :return: The combined values, combined systematic uncertainty, combined statistical uncertainty, and covariance matrix.
        :rtype: tuple of np.ndarray
        """
        if apply_bic_weighting:
            weights = cls.get_relative_bic_weights_for_sample(
                ppxf_fits_list,
                num_fixed_parameters=num_fixed_parameters,
                num_bootstrap_samples=num_bootstrap_samples,
                weight_threshold=weight_threshold,
            )
        else:
            weights = np.ones(len(ppxf_fits_list))

        if verbose:
            print(f"BIC weighting {'' if apply_bic_weighting else 'not'} applied")
            print("Weights:", weights / np.sum(weights))

        (
            combined_values,
            combined_systematic_uncertainty,
            combined_statistical_uncertainty,
            covariance,
        ) = cls.combine_weighted(values, uncertanties, weights)

        return (
            combined_values,
            combined_systematic_uncertainty,
            combined_statistical_uncertainty,
            covariance,
        )

    @classmethod
    def combine_weighted(cls, values, uncertanties, weights):
        """
        Combine the values using the weights.

        :param values: The values to combine.
        :type values: np.ndarray
        :param uncertanties: The uncertainties in the values.
        :type uncertanties: np.ndarray
        :param weights: The weights to use for the combination.
        :type weights: np.ndarray
        :return: The combined values, combined systematic uncertainty, combined statistical uncertainty, and covariance matrix.
        """
        sum_w2 = np.sum(weights**2)
        sum_w = np.sum(weights)
        w = weights[:, np.newaxis]

        if values.ndim == 1:
            values = values[:, np.newaxis]
            uncertanties = uncertanties[:, np.newaxis]

        combined_values = np.sum(w * values, axis=0) / sum_w

        combined_systematic_uncertainty = np.sqrt(
            np.sum(w * (values - combined_values[np.newaxis, :]) ** 2, axis=0)
            / (sum_w - sum_w2 / sum_w)
        )

        combined_statistical_uncertainty = np.sqrt(
            np.sum(w * uncertanties**2, axis=0) / sum_w
        )

        if values.shape[1] > 1:
            covariance = np.zeros((len(combined_values), len(combined_values)))

            for i in range(covariance.shape[0]):
                for j in range(covariance.shape[0]):
                    covariance[i, j] = np.sum(
                        weights
                        * (values[:, i] - combined_values[i])
                        * (values[:, j] - combined_values[j])
                    ) / (sum_w - sum_w2 / sum_w)

                    if i == j:
                        covariance[i, j] += combined_statistical_uncertainty[i] ** 2
        else:
            covariance = None
        return (
            combined_values,
            combined_systematic_uncertainty,
            combined_statistical_uncertainty,
            covariance,
        )

    @staticmethod
    def calculate_weights_from_bic(delta_bic, sigma_delta_bic):
        """
        Calculate the relative BIC weight after accounting for the uncetainty.

        :param delta_bic: The difference in BIC values between the model and the best model.
        :type delta_bic: float
        :param sigma_delta_bic: The uncertainty in the delta_BIC value.
        :type sigma_delta_bic: float
        :return: The relative weight of the model.
        :rtype: float
        """
        integral_1 = ndtr(-delta_bic / sigma_delta_bic)
        integral_2 = ndtr(delta_bic / sigma_delta_bic - sigma_delta_bic / 2)
        exp_factor = (sigma_delta_bic**2 / 8) - (delta_bic / 2)

        if integral_2 == 0.0:
            integral2_multiplied = 0.0
        else:
            integral2_multiplied = np.exp(exp_factor + np.log(integral_2))

        weight = integral_1 + integral2_multiplied

        return weight

    @staticmethod
    def boost_noise(spectra, boost_factor, boosting_mask=None):
        """
        Boost the noise in the spectra.

        :param  spectra: The spectra to boost the noise in.
        :type spectra: `Spectra` or a child class
        :param boost_factor: The factor to boost the noise by.
        :type boost_factor: float
        :param boosting_mask: The mask to apply the boosting to.
        :type boosting_mask: np.ndarray
        :return spectra: The spectra with the boosted noise.
        :rtype: `Spectra` or a child class
        """
        noise_boosted_spectra = deepcopy(spectra)
        if boosting_mask is None:
            boosting_mask = np.ones_like(spectra.wavelengths, dtype=bool)

        if noise_boosted_spectra.noise is not None:
            noise_boosted_spectra.noise[boosting_mask] *= boost_factor

        if noise_boosted_spectra.covariance is not None:
            noise_boosted_spectra.covariance[boosting_mask] *= boost_factor
            noise_boosted_spectra.covariance[:, boosting_mask] *= boost_factor

        return noise_boosted_spectra
