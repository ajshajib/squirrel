"""This module contains the class to store data products in 3D datacube or 2D detector
image."""

from copy import deepcopy


__author__ = "ajshajib"


class Data(object):
    """A class to store spectroscopic data."""

    def __init__(
        self,
        wavelengths,
        spectra,
        wavelength_unit,
        fwhm,
        z_lens,
        z_source,
        spectra_unit="arbitrary",
        mask=None,
        noise=None,
    ):
        """
        :param wavelengths: wavelengths of the spectra in observer frame
        :param spectra: spectra of the data
        :param wavelength_unit: unit of the wavelengths
        :param fwhm: full width at half maximum of the data. Needs to be in the same unit as the wavelengths
        :param spectra_unit: unit of the spectra
        :param mask: mask of the data
        :param noise: noise of the data
        :param z_lens: lens redshift
        :param z_source: source redshift
        """
        self._wavelengths = deepcopy(wavelengths)
        self._original_wavelengths = wavelengths
        self._spectra = deepcopy(spectra)
        self._original_spectra = spectra
        self._spectra_unit = spectra_unit
        self._wavelength_unit = wavelength_unit
        self._spectra_state = "original"
        self._wavelengths_frame = "observer"
        self._wavelengths_state = "original"

        self._z_lens = z_lens
        self._z_source = z_source

        self.fwhm = fwhm
        self.restframe_fwhm = self.fwhm / (1 + self.z_lens)

        self._mask = mask
        self._noise = noise

        self._velocity_scale = None

    @property
    def spectra(self):
        """Return the spectra of the data."""
        if hasattr(self, "_spectra"):
            return self._spectra

    @spectra.setter
    def spectra(self, spectra):
        """Set the spectra of the data."""
        self._spectra = spectra

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
    def wavelength_state(self):
        """Return the state of the wavelengths."""
        if hasattr(self, "_wavelengths_state"):
            return self._wavelengths_state

    @wavelength_state.setter
    def wavelength_state(self, state):
        """Set the state of the wavelengths."""
        self._wavelengths_state = state

    @property
    def wavelengths_frame(self):
        """Return the frame of the wavelengths."""
        if hasattr(self, "_wavelengths_frame"):
            return self._wavelengths_frame

    @wavelengths_frame.setter
    def wavelengths_frame(self, frame):
        """Set the frame of the wavelengths."""
        self._wavelengths_frame = frame

    @property
    def spectra_unit(self):
        """Return the unit of the spectra."""
        if hasattr(self, "_spectra_unit"):
            return self._spectra_unit

    @property
    def spectra_state(self):
        """Return the state of the spectra."""
        if hasattr(self, "_spectra_state"):
            return self._spectra_state

    @spectra_state.setter
    def spectra_state(self, state):
        """Set the state of the spectra."""
        self._spectra_state = state

    @property
    def wavelength_unit(self):
        """Return the unit of the wavelengths."""
        if hasattr(self, "_wavelength_unit"):
            return self._wavelength_unit

    @property
    def mask(self):
        """Return the mask of the data."""
        if hasattr(self, "_mask"):
            return self._mask

    @property
    def noise(self):
        """Return the noise of the data."""
        if hasattr(self, "_noise"):
            return self._noise

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


class Datacube(Data):
    """A class to store in 3D IFU datacubes."""

    def __init__(
        self,
        wavelengths,
        spectra,
        wavelength_unit,
        spectra_unit=None,
        mask=None,
        noise=None,
        z_lens=None,
        z_source=None,
    ):
        """
        :param wavelengths: wavelengths of the data
        :param spectra: spectra of the data
        :param wavelength_unit: unit of the wavelengths
        :param spectra_unit: unit of the spectra
        :param mask: mask of the data
        :param noise: noise of the data
        :param z_lens: lens redshift
        :param z_source: source redshift
        """
        super(Datacube, self).__init__(
            wavelengths,
            spectra,
            wavelength_unit,
            spectra_unit,
            mask,
            noise,
            z_lens,
            z_source,
        )
