"""This module contains the class to wrap the pPXF package for kinematic analysis."""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import ppxf.ppxf_util as ppxf_util
from ppxf.ppxf import ppxf
from vorbin.voronoi_2d_binning import voronoi_2d_binning

from .data import VoronoiBinnedSpectra


class Pipeline(object):
    """A class to wrap the pPXF package for kinematic analysis."""

    _speed_of_light = 299792.458  # speed of light in km/s

    @classmethod
    def log_rebin(cls, data, velocity_scale=None):
        """Rebin the data to log scale.

        :param data: data to rebin
        :type data: `Data` class
        """
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

    @classmethod
    def run_ppxf(
        cls,
        data,
        template,
        velocity_dispersion_guess=250.0,
        degree=2,
        velocity_scale_ratio=2,
        background_template=None,
        spectra_indices=None,
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
        :param velocity_scale_ratio: ratio of the velocity scale to the velocity dispersion
        :type velocity_scale_ratio: float
        :param background_spectra: background spectra to fit
        :type background_spectra: `Data` class
        :param spectra_indices: indices of the spectra to fit, used for datacubes or radially binned spectra
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
            moments=2,
            goodpixels=None,
            lam=data.wavelengths,
            lam_temp=template.wavelengths,
            degree=degree,
            velscale_ratio=velocity_scale_ratio,
            sky=background_template.flux if background_template else None,
        )

        return ppxf_fit

    @classmethod
    def voronoi_bin(
        cls,
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

        bin_num, x_node, y_node, bin_cen_x, bin_cen_y, snr, num_pixels, scale = (
            voronoi_2d_binning(
                xx_coordinates_masked,
                yy_coordinates_masked,
                snr_per_wavelength_unit_masked,
                np.ones_like(snr_per_wavelength_unit_masked),
                target_snr,
                plot=plot,
                quiet=quiet,
            )
        )

        if plot:
            plt.tight_layout()

        voronoi_binned_flux = np.zeros(
            (datacube.flux.shape[0], int(np.max(bin_num)) + 1)
        )
        voronoi_binned_noise = np.zeros_like(voronoi_binned_flux)

        # for i in range(voronoi_bins.shape[0]):
        for x, y, n_bin in zip(xx_pixels_masked, yy_pixels_masked, bin_num):
            voronoi_binned_flux[:, n_bin] += datacube.flux[:, y, x]
            voronoi_binned_noise[:, n_bin] += datacube.noise[:, y, x] ** 2

        voronoi_binned_noise = np.sqrt(voronoi_binned_noise)

        voronoi_binned_spectra = VoronoiBinnedSpectra(
            datacube.wavelengths,
            voronoi_binned_flux,
            datacube.wavelength_unit,
            datacube.fwhm,
            datacube.z_lens,
            datacube.z_source,
            bin_cen_x,
            bin_cen_y,
            bin_num,
            datacube.flux_unit,
            voronoi_binned_noise,
        )

        voronoi_binned_spectra.spectra_modifications = deepcopy(
            datacube.spectra_modifications
        )
        voronoi_binned_spectra.wavelengths_frame = deepcopy(datacube.wavelengths_frame)

        return voronoi_binned_spectra
