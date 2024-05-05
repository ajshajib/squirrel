"""This module contains the class to wrap the pPXF package for kinematic analysis."""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import ppxf.ppxf_util as ppxf_util
from ppxf.ppxf import ppxf
import ppxf.sps_util as sps_util
from vorbin.voronoi_2d_binning import voronoi_2d_binning

from .data import VoronoiBinnedSpectra
from .template import Template


class Pipeline(object):
    """A class to wrap the pPXF package for kinematic analysis."""

    _speed_of_light = 299792.458  # speed of light in km/s

    @staticmethod
    def log_rebin(data, velocity_scale=None):
        """Rebin the data to log scale.

        :param data: data to rebin
        :type data: `Data` class
        """
        if "log_rebinned" in data.spectra_modifications:
            raise ValueError("Data has already been log rebinned.")

        wavelength_range = data.wavelengths[[0, -1]]

        rebinned_spectra, log_rebinned_wavelength, velocity_scale = ppxf_util.log_rebin(
            wavelength_range, data.flux, velscale=velocity_scale
        )

        # this may be problematic, just using this line as a placeholder for now until further checks
        rebinned_variance, _, _ = ppxf_util.log_rebin(
            wavelength_range, data.noise**2, velscale=velocity_scale
        )

        data.flux = rebinned_spectra
        data.noise = np.sqrt(rebinned_variance)
        data.wavelengths = np.exp(log_rebinned_wavelength)
        data.velocity_scale = velocity_scale
        data.spectra_modifications += ["log_rebinned"]

    @staticmethod
    def voronoi_bin(
        datacube,
        target_snr,
        min_wavelength_for_snr,
        max_wavelength_for_snr,
        max_radius,
        min_snr_per_spaxel=1.0,
        plot=False,
        quiet=True,
    ):
        """Perform the Voronoi binning.

        :param datacube: datacube to bin
        :type datacube: `DataCube` class
        :param snr_target_per_angstrom: target S/N per wavelength unit for each bin
        :type snr_target_per_angstrom: float
        :param min_wavelength_for_snr: minimum wavelength for S/N calculation, in the unit of `datacube.wavelengths`
        :type min_wavelength_for_snr: float
        :param max_wavelength_for_snr: maximum wavelength for S/N calculation
        :type max_wavelength_for_snr: float
        :param max_radius: maximum radius for binning, in the unit of in `datacube.x_coordinates`
        :type max_radius: float
        :param min_snr_per_spaxel: minimum S/N per spaxel to include in the binning
        :type min_snr_per_spaxel: float
        :param plot: plot the results
        :type plot: bool
        :param quiet: suppress the output
        :type quiet: bool
        :return: Voronoi binned spectra
        :rtype: `VoronoiBinnedSpectra` class
        """

        clipped_datacube = deepcopy(datacube)
        clipped_datacube.clip(
            wavelength_min=min_wavelength_for_snr, wavelength_max=max_wavelength_for_snr
        )

        snr_per_wavelength_unit = np.median(
            clipped_datacube.flux / clipped_datacube.noise, axis=0
        ) / (clipped_datacube.wavelengths[1] - clipped_datacube.wavelengths[0])

        radius = np.sqrt(datacube.x_coordinates**2 + datacube.y_coordinates**2)

        snr_mask = (snr_per_wavelength_unit > min_snr_per_spaxel) & (
            radius < max_radius
        )

        x_pixels = np.arange(datacube.flux.shape[2])
        y_pixels = np.arange(datacube.flux.shape[1])
        xx_pixels, yy_pixels = np.meshgrid(x_pixels, y_pixels)

        xx_pixels_masked = xx_pixels[snr_mask]
        yy_pixels_masked = yy_pixels[snr_mask]

        xx_coordinates_masked = datacube.x_coordinates[snr_mask]
        yy_coordinates_masked = datacube.y_coordinates[snr_mask]

        snr_per_wavelength_unit_masked = snr_per_wavelength_unit[snr_mask]

        (
            bin_numbers,
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
            snr_per_wavelength_unit_masked,
            np.ones_like(snr_per_wavelength_unit_masked),
            target_snr,
            plot=plot,
            quiet=quiet,
        )

        if plot:
            plt.tight_layout()

        voronoi_binned_flux = np.zeros(
            (datacube.flux.shape[0], int(np.max(bin_numbers)) + 1)
        )
        voronoi_binned_noise = np.zeros_like(voronoi_binned_flux)

        # for i in range(voronoi_bins.shape[0]):
        for x, y, n_bin in zip(xx_pixels_masked, yy_pixels_masked, bin_numbers):
            voronoi_binned_flux[:, n_bin] += datacube.flux[:, y, x]
            voronoi_binned_noise[:, n_bin] += datacube.noise[:, y, x] ** 2

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
            bin_numbers=bin_numbers,
            x_pixels_of_bins=xx_pixels_masked,
            y_pixels_of_bins=yy_pixels_masked,
            flux_unit=datacube.flux_unit,
            noise=voronoi_binned_noise,
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
        wavelength_range_extend_factor=1.2,
    ):
        """Get the template object created for a stellar template library. The `library_path` should point to a `numpy.savez()` file containing the following arrays for a given SPS models library, like FSPS, Miles, GALEXEV, BPASS. This file will be sent to `ppxf.sps_util.sps_lib()`. See the documentation of that function for the format of the file.
        The EMILES, FSPS, GALEXEV libraries are available at https://github.com/micappe/ppxf_data.

        :param library_path: path to the library
        :type library_path: str
        :param spectra: log rebinned spectra
        :type spectra: `Data` class
        :param velocity_scale_ratio: velocity scale ratio for the template
        :type velocity_scale_ratio: float
        :param wavelength_range_extend_factor: factor to extend the wavelength range
        :type wavelength_range_extend_factor: float
        :return: template
        :rtype: `Template` class
        """
        assert spectra.wavelength_unit == "AA", "Wavelength unit must be in Angstrom."
        assert (
            "log_rebinned" in spectra.spectra_modifications
        ), "Data must be log rebinned."

        wavelength_range_templates = (
            spectra.wavelengths[0] / wavelength_range_extend_factor,
            spectra.wavelengths[-1] * wavelength_range_extend_factor,
        )

        # template library will be sampled at data resolution times the velscale_ratio in the given wavelength range
        sps = sps_util.sps_lib(
            library_path,
            spectra.velocity_scale / velocity_scale_ratio,
            spectra.fwhm,
            wave_range=wavelength_range_templates,
            norm_range=[spectra.wavelengths[0], spectra.wavelengths[-1]],
        )

        template_fluxes = sps.templates.reshape(sps.templates.shape[0], -1)
        templates_wavelengths = sps.lam_temp

        template = Template(
            templates_wavelengths,
            template_fluxes,
            wavelength_unit="AA",
            fwhm=spectra.fwhm,
        )

        return template

    @staticmethod
    def run_ppxf(
        data,
        template,
        velocity_dispersion_guess=250.0,
        degree=2,
        moment=2,
        velocity_scale_ratio=2,
        background_template=None,
        spectra_indices=None,
        quiet=True,
    ):
        """Perform the kinematic analysis using pPXF.

        :param data: data to analyze
        :type data: `Data` class
        :param template_library: library of templates
        :type template_library: `TemplateLibrary` class
        :param velocity_dispersion_guess: initial guess for the velocity dispersion
        :type velocity_dispersion_guess: float
        :param degree: degree of the additive polynomial
        :type degree: int
        :param moment: order of the Gauss-Hermite series
        :type moment: int
        :param velocity_scale_ratio: ratio of the velocity scale to the velocity dispersion
        :type velocity_scale_ratio: float
        :param background_template: background spectra to fit
        :type background_template: `Data` class
        :param spectra_indices: indices of the spectra to fit, used for datacubes or binned spectra
        :type spectra_indices: list of int or int
        :return: pPXF fit
        :rtype: `ppxf` class
        """
        initial_guess = [
            0.0,
            velocity_dispersion_guess,
        ]  # (km/s), starting guess for [V, sigma]

        if spectra_indices is not None:
            if isinstance(spectra_indices, int) and data.flux.ndim == 2:
                assert (
                    data.flux.ndim == 2
                ), f"Spectra indices must be an integer for spectra with {data.spectra.ndim} dimensions."
                flux = data.flux[:, spectra_indices]
                noise = data.noise[:, spectra_indices]
                print(flux.shape, noise.shape)
                print()
            elif (
                isinstance(spectra_indices, list)
                and len(spectra_indices) == data.flux.ndim - 1
                and data.flux.ndim == 3
            ):
                flux = data.flux[:, spectra_indices[0], spectra_indices[1]]
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
            noise = data.noise

        ppxf_fit = ppxf(
            templates=template.flux,
            galaxy=flux,
            noise=noise,
            velscale=data.velocity_scale,
            start=initial_guess,
            plot=False,
            moments=moment,
            goodpixels=None,
            lam=data.wavelengths,
            lam_temp=template.wavelengths,
            degree=degree,
            velscale_ratio=velocity_scale_ratio,
            sky=background_template.flux if background_template else None,
            quiet=quiet,
        )

        return ppxf_fit

    @classmethod
    def run_ppxf_on_binned_spectra(
        cls,
        binned_spectra,
        template,
        velocity_dispersion_guess=250.0,
        degree=2,
        moment=2,
        velocity_scale_ratio=2,
        background_template=None,
        spectra_indices=None,
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
                velocity_dispersion_guess=velocity_dispersion_guess,
                degree=degree,
                velocity_scale_ratio=velocity_scale_ratio,
                background_template=background_template,
                spectra_indices=i,
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
