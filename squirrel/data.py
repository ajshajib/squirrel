"""This module contains the class to store data products in 3D datacube or 2D detector
image."""

from copy import deepcopy
import numpy as np

__author__ = "ajshajib"


class Spectra(object):
    """A class to store spectroscopic data."""

    def __init__(
        self,
        wavelengths,
        flux,
        wavelength_unit,
        fwhm,
        z_lens,
        z_source,
        flux_unit="arbitrary",
        noise=None,
    ):
        """
        :param wavelengths: wavelengths of the spectra in observer frame
        :type wavelengths: numpy.ndarray
        :param flux: flux of the data
        :type flux: numpy.ndarray
        :param wavelength_unit: unit of the wavelengths
        :type wavelength_unit: str
        :param fwhm: full width at half maximum of the data. Needs to be in the same unit as the wavelengths
        :type fwhm: float
        :param flux_unit: unit of the flux
        :type flux_unit: str
        :param mask: mask of the data
        :type mask: numpy.ndarray
        :param noise: noise of the data
        :type noise: numpy.ndarray
        :param z_lens: lens redshift
        :type z_lens: float
        :param z_source: source redshift
        :type z_source: float
        """
        self._wavelengths = deepcopy(wavelengths)
        self._original_wavelengths = wavelengths
        self._flux = deepcopy(flux)
        self._original_flux = flux
        self._flux_unit = flux_unit
        self._wavelength_unit = wavelength_unit
        self._spectra_modifications = []
        self._wavelengths_frame = "observed"

        self._z_lens = z_lens
        self._z_source = z_source

        self.fwhm = fwhm
        self.restframe_fwhm = self.fwhm / (1 + self.z_lens)

        self._noise = deepcopy(noise)
        self._original_noise = noise

        self._velocity_scale = None

    @property
    def flux(self):
        """Return the flux of the data."""
        if hasattr(self, "_flux"):
            return self._flux

    @flux.setter
    def flux(self, flux):
        """Set the flux of the data."""
        self._flux = flux

    @property
    def original_flux(self):
        """Return the original flux of the data."""
        if hasattr(self, "_original_flux"):
            return self._original_flux

    @property
    def original_wavelengths(self):
        """Return the original wavelengths of the data."""
        if hasattr(self, "_original_wavelengths"):
            return self._original_wavelengths

    @property
    def wavelengths(self):
        """Return the wavelengths of the data."""
        if hasattr(self, "_wavelengths"):
            return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, wavelengths):
        """Set the wavelengths of the data."""
        self._wavelengths = wavelengths

    @property
    def wavelengths_frame(self):
        """Return the frame of the wavelengths.

        Possible frames are 'observed', 'lens', 'source', or 'z={`redshift`}'.
        """
        if hasattr(self, "_wavelengths_frame"):
            return self._wavelengths_frame

    @wavelengths_frame.setter
    def wavelengths_frame(self, frame):
        """Set the frame of the wavelengths."""
        self._wavelengths_frame = frame

    @property
    def flux_unit(self):
        """Return the unit of the flux."""
        if hasattr(self, "_flux_unit"):
            return self._flux_unit

    @property
    def spectra_modifications(self):
        """Return the state of the spectra.

        Possible states are 'original', 'rebinned', or 'log_rebinned'.
        """
        if hasattr(self, "_spectra_modifications"):
            return self._spectra_modifications

    @spectra_modifications.setter
    def spectra_modifications(self, state):
        """Set the state of the spectra."""
        self._spectra_modifications = state

    @property
    def wavelength_unit(self):
        """Return the unit of the wavelengths."""
        if hasattr(self, "_wavelength_unit"):
            return self._wavelength_unit

    @property
    def noise(self):
        """Return the noise of the data."""
        if hasattr(self, "_noise"):
            return self._noise

    @noise.setter
    def noise(self, noise):
        """Set the noise of the data."""
        self._noise = noise

    @property
    def original_noise(self):
        """Return the original noise of the data."""
        if hasattr(self, "_original_noise"):
            return self._original_noise

    @property
    def z_lens(self):
        """Return the lens redshift."""
        if hasattr(self, "_z_lens"):
            return self._z_lens

    @property
    def z_source(self):
        """Return the source redshift."""
        if hasattr(self, "_z_source"):
            return self._z_source

    @property
    def velocity_scale(self):
        """Return the velocity scale of the data."""
        if hasattr(self, "_velocity_scale"):
            return self._velocity_scale

    @velocity_scale.setter
    def velocity_scale(self, velocity_scale):
        """Set the velocity scale of the data."""
        self._velocity_scale = velocity_scale

    def deredshift(self, redshift=None, target_frame=None):
        """Deredshift the spectra.

        :param data: data to deredshift
        :type data: `Data` class
        :param redshift: redshift to deredshift the data to
        :type redshift: float
        :param target_frame: frame to deredshift the data to, "lens" or "source"
        :type target_frame: str
        """
        if redshift is None:
            if target_frame == "lens":
                redshift = self.z_lens
                self._wavelengths_frame = "lens frame"
            elif target_frame == "source":
                redshift = self.z_source
                self._wavelengths_frame = "source frame"
            else:
                raise ValueError(
                    "If redshift is not provided, frame must be either 'lens' or 'source'"
                )
        else:
            self._wavelengths_frame = f"z={redshift:.3f}"

        self._wavelengths = self._wavelengths / (1.0 + redshift)

    def reset(self):
        """Reset the data to the original state."""
        self._wavelengths = deepcopy(self._original_wavelengths)
        self._flux = deepcopy(self._original_flux)
        self._noise = deepcopy(self._original_noise)
        self._spectra_modifications = []
        self._wavelengths_frame = "observed"

    def clip(self, wavelength_min, wavelength_max):
        """Clip the data to the given wavelength range.

        :param wavelength_range: range of the wavelengths to clip
        :type wavelength_range: list
        """
        mask = (self._wavelengths >= wavelength_min) & (
            self._wavelengths <= wavelength_max
        )

        self._wavelengths = self._wavelengths[mask]
        if len(self.flux.shape) == 1:
            self._flux = self._flux[mask]
            if self._noise is not None:
                self._noise = self._noise[mask]
        else:
            self._flux = self._flux[mask, :]
            if self._noise is not None:
                self._noise = self._noise[mask, :]

        self._spectra_modifications += ["clipped"]


class Datacube(Spectra):
    """A class to store in 3D IFU datacubes."""

    def __init__(
        self,
        wavelengths,
        flux,
        wavelength_unit,
        fwhm,
        z_lens,
        z_source,
        center_pixel_x,
        center_pixel_y,
        coordinate_transform_matrix,
        flux_unit="arbitrary",
        noise=None,
    ):
        """
        :param wavelengths: wavelengths of the data
        :type wavelengths: numpy.ndarray
        :param flux: flux of the data
        :type flux: numpy.ndarray
        :param wavelength_unit: unit of the wavelengths
        :type wavelength_unit: str
        :param fwhm: full width at half maximum of the data
        :type fwhm: float
        :param spatial_pixel_size: size of the spatial pixels
        :type spatial_pixel_size: float
        :param z_lens: lens redshift
        :type z_lens: float
        :param z_source: source redshift
        :type z_source: float
        :param flux_unit: unit of the flux
        :type flux_unit: str
        :param noise: noise of the data
        :type noise: numpy.ndarray
        """
        super(Datacube, self).__init__(
            wavelengths=wavelengths,
            flux=flux,
            wavelength_unit=wavelength_unit,
            fwhm=fwhm,
            z_lens=z_lens,
            z_source=z_source,
            flux_unit=flux_unit,
            noise=noise,
        )

        self._center_pixel_x = center_pixel_x
        self._center_pixel_y = center_pixel_y

        x_pixels = np.arange(self._flux.shape[2]) - self._center_pixel_x
        y_pixels = np.arange(self._flux.shape[1]) - self._center_pixel_y

        xx_pixels, yy_pixels = np.meshgrid(x_pixels, y_pixels)

        transformed_coordinates = np.dot(
            coordinate_transform_matrix,
            np.array([xx_pixels.flatten(), yy_pixels.flatten()]),
        )
        self._x_coordinates = transformed_coordinates[0].reshape(self._flux.shape[1:])
        self._y_coordinates = transformed_coordinates[1].reshape(self._flux.shape[1:])

    @property
    def center_pixel_x(self):
        """Return the x coordinate of the center pixel."""
        if hasattr(self, "_center_pixel_x"):
            return self._center_pixel_x

    @property
    def center_pixel_y(self):
        """Return the y coordinate of the center pixel."""
        if hasattr(self, "_center_pixel_y"):
            return self._center_pixel_y

    @property
    def x_coordinates(self):
        """Return the x coordinates of the data."""
        if hasattr(self, "_x_coordinates"):
            return self._x_coordinates

    @property
    def y_coordinates(self):
        """Return the y coordinates of the data."""
        if hasattr(self, "_y_coordinates"):
            return self._y_coordinates


class VoronoiBinnedSpectra(Spectra):
    """A class to store binned spectra."""

    def __init__(
        self,
        wavelengths,
        flux,
        wavelength_unit,
        fwhm,
        z_lens,
        z_source,
        x_coordinates,
        y_coordinates,
        bin_num,
        flux_unit="arbitrary",
        noise=None,
    ):
        """
        :param wavelengths: wavelengths of the data
        :type wavelengths: numpy.ndarray
        :param flux: flux of the data
        :type flux: numpy.ndarray
        :param wavelength_unit: unit of the wavelengths
        :type wavelength_unit: str
        :param fwhm: full width at half maximum of the data
        :type fwhm: float
        :param z_lens: lens redshift
        :type z_lens: float
        :param z_source: source redshift
        :type z_source: float
        :param x_coordinates: x coordinates of the data
        :type x_coordinates: numpy.ndarray
        :param y_coordinates: y coordinates of the data
        :type y_coordinates: numpy.ndarray
        :param bin_num: bin number of the data
        :type bin_num: numpy.ndarray
        :param flux_unit: unit of the flux
        :type flux_unit: str
        :param noise: noise of the data
        :type noise: numpy.ndarray
        """
        super(VoronoiBinnedSpectra, self).__init__(
            wavelengths=wavelengths,
            flux=flux,
            wavelength_unit=wavelength_unit,
            fwhm=fwhm,
            z_lens=z_lens,
            z_source=z_source,
            flux_unit=flux_unit,
            noise=noise,
        )

        self._x_coordinates = x_coordinates
        self._y_coordinates = y_coordinates
        self._bin_num = bin_num

    @property
    def x_coordinates(self):
        """Return the x coordinates of the data."""
        if hasattr(self, "_x_coordinates"):
            return self._x_coordinates

    @property
    def y_coordinates(self):
        """Return the y coordinates of the data."""
        if hasattr(self, "_y_coordinates"):
            return self._y_coordinates

    @property
    def bin_num(self):
        """Return the bin number of the data."""
        if hasattr(self, "_bin_num"):
            return self._bin_num


class RadiallyBinnedSpectra(Spectra):
    """A class to store radially binned spectra."""

    def __init__(
        self,
        wavelengths,
        flux,
        wavelength_unit,
        fwhm,
        z_lens,
        z_source,
        bin_radii,
        flux_unit="arbitrary",
        noise=None,
    ):
        """
        :param wavelengths: wavelengths of the data
        :type wavelengths: numpy.ndarray
        :param flux: flux of the data
        :type flux: numpy.ndarray
        :param wavelength_unit: unit of the wavelengths
        :type wavelength_unit: str
        :param fwhm: full width at half maximum of the data
        :type fwhm: float
        :param z_lens: lens redshift
        :type z_lens: float
        :param z_source: source redshift
        :type z_source: float
        :param bin_radii: radial edges of the bins, starting with the inner edge of the first bin. The first value should be zero if the first bin is a circle.
        :type bin_radii: numpy.ndarray
        :param flux_unit: unit of the flux
        :type flux_unit: str
        :param noise: noise of the data
        :type noise: numpy.ndarray
        """
        assert (
            len(bin_radii) - 1 == flux.shape[1]
        ), "Number of bins must match the number of spectra."

        super(RadiallyBinnedSpectra, self).__init__(
            wavelengths=wavelengths,
            flux=flux,
            wavelength_unit=wavelength_unit,
            fwhm=fwhm,
            z_lens=z_lens,
            z_source=z_source,
            flux_unit=flux_unit,
            noise=noise,
        )

        self._bin_radii = bin_radii

    @property
    def bin_radii(self):
        """Return the radii of the bins."""
        if hasattr(self, "_bin_radii"):
            return self._bin_radii
