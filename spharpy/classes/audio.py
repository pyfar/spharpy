from pyfar import Signal
from spharpy.spherical import renormalize, change_channel_convention
import numpy as np


class SphericalHarmonicSignal(Signal):
    """Create audio object with spherical harmonics coefficients in time or
    frequency domain.

    Objects of this class contain spherical harmonics coefficients which are
    directly convertible between time and frequency domain (equally spaced
    samples and frequency bins), the channel conventions ACN and FUMA, as
    well as the normalizations N3D, SN3D, or MaxN, see [#]_. The definition of
    the spherical harmonics basis functions is based on the scipy convention
    which includes the Condon-Shortley phase, [#]_, [#]_.


    Parameters
    ----------
    data : ndarray, double
        Raw data of the spherical harmonics signal in the time or
        frequency domain. The data should have at least 3 dimensions,
        according to the 'C' memory layout, e.g. data of
        ``shape = (1, 4, 1024)`` has 1 channel with 4 spherical harmonic
        coefficients with 1024 samples or frequency
        bins each. Time data is converted to ``float``. Frequency is
        converted to ``complex`` and must be provided as single
        sided spectra, i.e., for all frequencies between 0 Hz and
        half the sampling rate.
    sampling_rate : double
        Sampling rate in Hz
    basis_type : str
        Type of spherical harmonic basis, either ``'complex'`` or
        ``'real'``.
    normalization : str
        Normalization convention, either ``'n3d'``, ``'maxN'`` or
        ``'sn3d'``. (maxN is only supported up to 3rd order)
    channel_convention : str
        Channel ordering convention, either ``'acn'`` or ``'fuma'``.
        (FuMa is only supported up to 3rd order)
    condon_shortley : bool
        Flag to indicate if the Condon-Shortley phase term is included
        (``True``) or not (``False``).
    n_samples : int, optional
        Number of time domain samples. Required if domain is ``'freq'``.
        The default is ``None``, which assumes an even number of samples
        if the data is provided in the frequency domain.
    domain : ``'time'``, ``'freq'``, optional
        Domain of data. The default is ``'time'``
    fft_norm : str, optional
        The normalization of the Discrete Fourier Transform (DFT). Can be
        ``'none'``, ``'unitary'``, ``'amplitude'``, ``'rms'``, ``'power'``,
        or ``'psd'``. See :py:func:`~pyfar.dsp.fft.normalization` and [#]_
        for more information. The default is ``'none'``, which is typically
        used for energy signals, such as impulse responses.
    comment : str
        A comment related to `data`. The default is ``None``.
    is_complex : bool, optional
        Specifies if the underlying time domain data are complex
        or real-valued. If ``True`` and `domain` is ``'time'``, the
        input data will be cast to complex. The default is ``False``.

    References
    ----------
    .. [#] F. Zotter, M. Frank, "Ambisonics A Practical 3D Audio Theory
            for Recording, Studio Production, Sound Reinforcement, and
            Virtual Reality", (2019), Springer-Verlag
    .. [#] B. Rafely, "Fundamentals of Spherical Array Processing", (2015),
            Springer-Verlag
    .. [#] E.G. Williams, "Fourier Acoustics", (1999), Academic Press

    """
    def __init__(
            self,
            data,
            sampling_rate,
            basis_type,
            normalization,
            channel_convention,
            condon_shortley,
            n_samples=None,
            domain='time',
            fft_norm='none',
            comment="",
            is_complex=False):
        """
        Create SphericalHarmonicSignal with data, and sampling rate.
        """
        # check dimensions
        if len(data.shape) < 3:
            raise ValueError("Invalid number of dimensions. Data should have "
                             "at least 3 dimensions.")

        # set n_max
        n_max = np.sqrt(data.shape[-2])-1
        if n_max - int(n_max) != 0:
            raise ValueError("Invalid number of SH channels: "
                             f"{data.shape[-2]}. It must match (n_max + 1)^2.")
        self._n_max = n_max

        # set basis_type
        if basis_type not in ["complex", "real"]:
            raise ValueError("Invalid basis type, only "
                             "'complex' and 'real' are supported")
        self._basis_type = basis_type

        # set normalization
        if normalization not in ["n3d", "maxN", "sn3d"]:
            raise ValueError("Invalid normalization, has to be 'sn3d', "
                             f"'n3d', or 'maxN', but is {normalization}")
        self._normalization = normalization

        # set channel_convention
        if channel_convention not in ["acn", "fuma"]:
            raise ValueError("Invalid channel convention, has to be 'acn' "
                             f"or 'fuma', but is {channel_convention}")
        self._channel_convention = channel_convention

        # set Condon Shortley
        if not isinstance(condon_shortley, bool):
            raise ValueError("Condon_shortley has to be a bool.")
        self._condon_shortley = condon_shortley

        Signal.__init__(self, data, sampling_rate=sampling_rate,
                        n_samples=n_samples, domain=domain, fft_norm=fft_norm,
                        comment=comment, is_complex=is_complex)

    @property
    def n_max(self):
        """Get the maximum spherical harmonic order."""
        return self._n_max

    @property
    def basis_type(self):
        """Get the type of the spherical harmonic basis."""
        return self._basis_type

    @property
    def normalization(self):
        """
        Get or set and apply the normalization of the spherical harmonic
        coefficients.
        """
        return self._normalization

    @normalization.setter
    def normalization(self, value):
        """
        Get or set and apply the normalization of the spherical harmonic
        coefficients.
        """
        if self.normalization is not value:
            self._data = renormalize(self._data, self.channel_convention,
                                     self.normalization, value, axis=-2)
            self._normalization = value

    @property
    def condon_shortley(self):
        """Get info whether to include the Condon-Shortley phase term."""
        return self._condon_shortley

    @property
    def channel_convention(self):
        """
        Get or set and apply the channel convention of the spherical harmonic
        coefficients.
        """
        return self._channel_convention

    @channel_convention.setter
    def channel_convention(self, value):
        """
        Get or set and apply the channel convention of the spherical harmonic
        coefficients.
        """
        if self.channel_convention is not value:
            self._data = change_channel_convention(self._data,
                                                   self.channel_convention,
                                                   value, axis=-2)
            self._channel_convention = value
