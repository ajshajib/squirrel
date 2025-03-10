"""This module contains the class to store data products in 3D datacube or 2D detector
image."""

from copy import deepcopy
import numpy as np


class Spectra(object):
    """A class to store spectroscopic data."""

    def __init__(
        self,
        wavelengths,
        flux,
        wavelength_unit,
        fwhm,
        z_lens=0.0,
        z_source=0.0,
        flux_unit="arbitrary",
        noise=None,
        covariance=None,
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
        :param z_lens: lens redshift
        :type z_lens: float
        :param z_source: source redshift
        :type z_source: float
        :param flux_unit: unit of the flux
        :type flux_unit: str
        :param noise: noise of the data
        :type noise: numpy.ndarray
        :param covariance: covariance of the data
        :type covariance: numpy.ndarray
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

        self._fwhm = deepcopy(fwhm)
        self._original_fwhm = fwhm

        self._noise = deepcopy(noise)
        self._original_noise = noise
        self._covariance = deepcopy(covariance)
        self._original_covariance = covariance

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
    def fwhm(self):
        """Return the full width at half maximum of the data."""
        if hasattr(self, "_fwhm"):
            return self._fwhm

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
    def covariance(self):
        """Return the covariance of the data."""
        if hasattr(self, "_covariance"):
            return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        """Set the covariance of the data."""
        self._covariance = covariance

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
        self._fwhm = self._fwhm / (1.0 + redshift)

    def reset(self):
        """Reset the data to the original state."""
        self._wavelengths = deepcopy(self._original_wavelengths)
        self._flux = deepcopy(self._original_flux)
        self._noise = deepcopy(self._original_noise)
        self._covariance = deepcopy(self._original_covariance)
        self._fwhm = deepcopy(self._original_fwhm)
        self._spectra_modifications = []
        self._wavelengths_frame = "observed"
        self._velocity_scale = None

    def clip(self, wavelength_min, wavelength_max):
        """Clip the data to the given wavelength range.

        :param wavelength_range: range of the wavelengths to clip
        :type wavelength_range: list
        """
        mask = (self._wavelengths >= wavelength_min) & (
            self._wavelengths <= wavelength_max
        )

        self._wavelengths = self._wavelengths[mask]

        self._flux = self._flux[mask, ...]
        if self._noise is not None:
            self._noise = self._noise[mask, ...]
        if self._covariance is not None:
            self._covariance = self._covariance[mask, mask, ...]

        self._spectra_modifications += ["clipped"]

    def _add(self, other, destination):
        """Add two spectra together.

        :param destination: destination spectra
        :type destination: Spectra
        :param spectra1: spectra to add
        :type spectra1: Spectra
        :param spectra2: spectra to add
        :type spectra2: Spectra
        """
        assert np.all(self.wavelengths == other.wavelengths), "Wavelengths must match."
        assert self.flux.shape == other.flux.shape, "Spectra must have the same shape."

        destination.flux = self.flux + other.flux
        destination.noise = None
        destination.covariance = None

        if self.noise is not None and other.noise is not None:
            destination.noise = np.sqrt(self.noise**2 + other.noise**2)
        if self.covariance is not None and other.covariance is not None:
            destination.covariance = self.covariance + other.covariance

    def __add__(self, other):
        """Add two spectra together.

        :param other: spectra to add
        :type other: Spectra
        :return: sum of the two spectra
        :rtype: Spectra
        """
        spectra = deepcopy(self)

        self._add(other, spectra)

        return spectra

    def __iadd__(self, other):
        """Add two spectra together in place.

        :param other: spectra to add
        :type other: Spectra
        :return: sum of the two spectra
        :rtype: Spectra
        """
        self._add(other, self)

        return self

    def _concat(self, other, destination):
        """Concatenate two spectra together.

        :param other: spectra to add
        :type other: Spectra
        """
        assert (
            self.wavelength_unit == other.wavelength_unit
        ), "Wavelength units must match."
        assert self.fwhm == other.fwhm, "FWHM must match."
        assert (
            self.wavelengths[-1] < other.wavelengths[0]
        ), "Wavelengths must be increasing without overlap."
        assert (
            self.spectra_modifications == other.spectra_modifications
        ), "Spectra modifications must match."

        destination.wavelengths = np.concatenate((self.wavelengths, other.wavelengths))
        destination.flux = np.concatenate((self.flux, other.flux), axis=0)
        destination.noise = None
        destination.covariance = None

        if self.noise is not None and other.noise is not None:
            destination.noise = np.concatenate((self.noise, other.noise), axis=0)
        if self.covariance is not None and other.covariance is not None:
            destination.covariance = np.zeros(
                (
                    self.covariance.shape[0] + other.covariance.shape[0],
                    self.covariance.shape[1] + other.covariance.shape[1],
                    self.covariance.shape[2],
                    self.covariance.shape[3],
                )
            )
            destination.covariance[
                : self.covariance.shape[0], : self.covariance.shape[1]
            ] = self.covariance
            destination.covariance[
                self.covariance.shape[0] :, self.covariance.shape[1] :
            ] = other.covariance

    def concat(self, other):
        """Concatenate two spectra together.

        :param other: spectra to concatenate
        :type other: Spectra
        """
        spectra = deepcopy(self)

        self._concat(other, spectra)

        return spectra

    def __and__(self, other):
        """Concatenate two spectra together.

        :param other: spectra to concatenate
        :type other: Spectra
        """
        return self.concat(other)

    def __iand__(self, other):
        """Concatenate two spectra together in place.

        :param other: spectra to concatenate
        :type other: Spectra
        """
        self._concat(other, self)

        return self


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
        covariance=None,
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
            covariance=covariance,
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

    def get_1d_spectra(self, x=None, y=None, mask=None):
        """Return the spectra at a given pixel, or summed within given a mask. If
        nothing is provided, the entire datacube will be summed over.

        :param x: x coordinate of the pixel
        :type x: int
        :param y: y coordinate of the pixel
        :type y: int
        :param slice: slice to sum the spectra
        :type slice: slice
        :param mask: mask to sum the spectra
        :type mask: numpy.ndarray
        :return: spectrum at the given pixel
        :rtype: numpy.ndarray
        """
        noise = None
        covariance = None

        if mask is not None:
            flux = np.nansum(self.flux * mask[None, :, :], axis=(1, 2))
            if self.noise is not None:
                noise = np.sqrt(
                    np.nansum(self.noise**2 * mask[None, :, :], axis=(1, 2))
                )
            if self.covariance is not None:
                covariance = np.nansum(
                    self.covariance * mask[None, None, :, :], axis=(2, 3)
                )
        elif x is not None and y is not None:
            flux = self.flux[:, y, x]
            if self.noise is not None:
                noise = self.noise[:, y, x]
            if self.covariance is not None:
                covariance = self.covariance[:, :, y, x]
        elif (x is None and y is not None) or (x is not None and y is None):
            raise ValueError("Both x and y must be provided if one of them is.")
        else:
            flux = np.nansum(self.flux, axis=(1, 2))
            if self.noise is not None:
                noise = np.sqrt(np.nansum(self.noise**2, axis=(1, 2)))
            if self.covariance is not None:
                covariance = np.nansum(self.covariance, axis=(2, 3))

        spectra = Spectra(
            wavelengths=self.wavelengths,
            flux=flux,
            wavelength_unit=self.wavelength_unit,
            fwhm=self.fwhm,
            z_lens=self.z_lens,
            z_source=self.z_source,
            flux_unit=self.flux_unit,
            noise=noise,
            covariance=covariance,
        )

        spectra.spectra_modifications = deepcopy(self.spectra_modifications)
        spectra.velocity_scale = deepcopy(self.velocity_scale)

        return spectra

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
        num_bins,
        x_pixels_of_bins,
        y_pixels_of_bins,
        flux_unit="arbitrary",
        noise=None,
        covariance=None,
        bin_center_x=None,
        bin_center_y=None,
        area=None,
        snr=None,
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
        :param x_coordinates: x coordinates of the original datacube's spatial pixels (2D)
        :type x_coordinates: numpy.ndarray
        :param y_coordinates: y coordinates of the original datacube's spatial pixels (2D)
        :type y_coordinates: numpy.ndarray
        :param bin_numbers: bin number of the data
        :type bin_numbers: numpy.ndarray
        :param x_pixels_of_bins: pixel_x values corresponding to `bin_numbers`
        :type x_pixels_of_bins: numpy.ndarray
        :param y_pixels_of_bins: pixel_y values corresponding to `bin_numbers`
        :type y_pixels_of_bins: numpy.ndarray
        :param flux_unit: unit of the flux
        :type flux_unit: str
        :param noise: noise of the data
        :type noise: numpy.ndarray
        :param covariance: covariance of the data
        :type covariance: numpy.ndarray
        :param bin_center_x: x coordinates of the bin centers
        :type bin_center_x: numpy.ndarray
        :param bin_center_y: y coordinates of the bin centers
        :type bin_center_y: numpy.ndarray
        :param area: area of the bins
        :type area: numpy.ndarray
        :param snr: signal-to-noise ratio of the bins
        :type snr: numpy.ndarray
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
            covariance=covariance,
        )

        self._x_coordinates = x_coordinates
        self._y_coordinates = y_coordinates
        self._bin_numbers = num_bins
        self._x_pixels_of_bins = x_pixels_of_bins
        self._y_pixels_of_bins = y_pixels_of_bins
        self._bin_center_x = bin_center_x
        self._bin_center_y = bin_center_y
        self._area = area
        self._snr = snr

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
    def bin_numbers(self):
        """Return the bin number of the data."""
        if hasattr(self, "_bin_numbers"):
            return self._bin_numbers

    @property
    def x_pixels_of_bins(self):
        """Return the x pixel values corresponding to the bin numbers."""
        if hasattr(self, "_x_pixels_of_bins"):
            return self._x_pixels_of_bins

    @property
    def y_pixels_of_bins(self):
        """Return the y pixel values corresponding to the bin numbers."""
        if hasattr(self, "_y_pixels_of_bins"):
            return self._y_pixels_of_bins

    @property
    def bin_center_x(self):
        """Return the x coordinates of the bin centers."""
        if hasattr(self, "_bin_center_x"):
            return self._bin_center_x

    @property
    def bin_center_y(self):
        """Return the y coordinates of the bin centers."""
        if hasattr(self, "_bin_center_y"):
            return self._bin_center_y

    @property
    def area(self):
        """Return the area of the bins."""
        if hasattr(self, "_area"):
            return self._area

    @property
    def snr(self):
        """Return the signal-to-noise ratio of the bins."""
        if hasattr(self, "_snr"):
            return self._snr

    def get_spaxel_map_with_bin_number(self):
        """Return vornoi bin mapping. -1 is masked pixel. Unmasked pixel start counting
        from 0.

        :return: 2D array with bin mapping
        :rtype: numpy.ndarray
        """
        bin_numbers = self.bin_numbers

        mapping = np.zeros(self.x_coordinates.shape, dtype=int) - 1

        x_pixels = self.x_pixels_of_bins
        y_pixels = self.y_pixels_of_bins

        for v, x, y in zip(bin_numbers, x_pixels, y_pixels):
            mapping[int(y)][int(x)] = v

        return mapping

    def get_single_spectra(self, bin_index):
        """Return the spectra of a single bin.

        :param bin_index: bin number
        :type bin_index: int
        :return: spectra of the bin
        :rtype: numpy.ndarray
        """
        if self.noise is not None:
            noise = self.noise[:, bin_index]
        else:
            noise = None

        if self.covariance is not None:
            covariance = self.covariance[:, :, bin_index]
        else:
            covariance = None

        spectra = Spectra(
            wavelengths=self.wavelengths,
            flux=self.flux[:, bin_index],
            wavelength_unit=self.wavelength_unit,
            fwhm=self.fwhm,
            z_lens=self.z_lens,
            z_source=self.z_source,
            flux_unit=self.flux_unit,
            noise=noise,
            covariance=covariance,
        )

        spectra.velocity_scale = deepcopy(self.velocity_scale)

        return spectra


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
        covariance=None,
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
        :param covariance: covariance of the data
        :type covariance: numpy.ndarray
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
            covariance=covariance,
        )

        self._bin_radii = bin_radii

    @property
    def bin_radii(self):
        """Return the radii of the bins."""
        if hasattr(self, "_bin_radii"):
            return self._bin_radii
