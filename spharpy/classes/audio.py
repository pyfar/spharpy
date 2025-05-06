from pyfar import Signal
from spharpy.spherical import renormalize, change_channel_convention
import numpy as np


class SphericalHarmonicSignal(Signal):
    """Create audio object with spherical harmonics coefficients in time or
    frequency domain.

    Objects of this class contain spherical harmonics coefficients which are
    directly convertible between time and frequency domain (equally spaced
    samples and frequency bins), the channel conventions `acn` and `fuma`, as
    well as the normalizations `n3d`, `sn3d`, or `maxn`. The default
    parameters for `basis_type`, `normalization`, and `channel_convention`
    corresponds to the AmbiX standard, see [#]_. The definition of the
    spherical harmonics basis functions is based on the scipy convention which
    includes the Condon-Shortley phase, [#]_, [#]_.


    Parameters
    ----------
    data : ndarray, double
        Raw data of the spherical harmonics signal in the time or
        frequency domain. The memory layout of data is 'C'. E.g.
        data of ``shape = (3, 4, 1024)`` has 3 channels with 4
        spherical harmonic coefficients with 1024 samples or frequency
        bins each. Time data is converted to ``float``. Frequency is
        converted to ``complex`` and must be provided as single
        sided spectra, i.e., for all frequencies between 0 Hz and
        half the sampling rate.
    sampling_rate : double
        Sampling rate in Hz
    n_max : int
        Maximum spherical harmonic order. Has to match the number of
        coefficients, such that the number of coefficients
        :math:`>= (n_{max} + 1) ^ 2`.
    basis_type : str
        Type of spherical harmonic basis, either ``'complex'`` or
        ``'real'``. The default is ``'real'``.
    normalization : str
        Normalization convention, either ``'n3d'``, ``'maxN'`` or
        ``'sn3d'``. The default is ``'n3d'``.
        (maxN is only supported up to 3rd order)
    channel_convention : str
        Channel ordering convention, either ``'acn'`` or ``'fuma'``.
        The default is ``'acn'``.
        (FuMa is only supported up to 3rd order)
    condon_shortley : bool or str, optional
        Whether to include the Condon-Shortley phase term. If ``True`` or
        ``'auto'``, Condon-Shortley is included, if ``False`` it is not
        included. The default is ``'auto'``.
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
            n_max,
            basis_type,
            normalization,
            channel_convention,
            condon_shortley='auto',
            n_samples=None,
            domain='time',
            fft_norm='none',
            comment="",
            is_complex=False):
        """
        Create SphericalHarmonicSignal with data, and sampling rate.
        """
        # set n_max
        if (n_max < 0) or (n_max % 1 != 0):
            raise ValueError("n_max must be a positive integer")
        if not data.shape[-2] >= (n_max + 1) ** 2:
            raise ValueError('Data has to few sh coefficients for '
                             f'{n_max=}. Highest possible n_max is '
                             f'{int(np.sqrt(data.shape[-2]))-1}')
        self._n_max = int(n_max)

        # set basis_type
        if basis_type == 'complex' and not is_complex:
            raise ValueError('Data are real-valued while '
                             'spherical harmonics bases are complex-valued.')

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
        if not isinstance(condon_shortley, bool) and condon_shortley != 'auto':
            raise ValueError(
                "Condon_shortley has to be a bool, or 'auto'.")

        if condon_shortley == 'auto' and basis_type == 'complex':
            self._condon_shortley = True
        elif condon_shortley == 'auto' and basis_type == 'real':
            self._condon_shortley = False

        Signal.__init__(self, data, sampling_rate=sampling_rate,
                        n_samples=n_samples, domain=domain, fft_norm=fft_norm,
                        comment=comment, is_complex=is_complex)

    @property
    def n_max(self):
        """the maximum spherical harmonic order."""
        return self._n_max

    @property
    def basis_type(self):
        """the type of the spherical harmonic basis (``'complex'`` or ``'real'``).
        """
        return self._basis_type

    @property
    def normalization(self):
        """the normalization of the spherical harmonic coefficients."""
        return self._normalization

    @normalization.setter
    def normalization(self, value):
        """set the normalization of the spherical harmonic coefficients."""
        if self.normalization is not value:
            self._data = renormalize(self._data, self.channel_convention,
                                     self.normalization, value, axis=-2)
            self._normalization = value

    @property
    def condon_shortley(self):
        """whether to include the Condon-Shortley phase term."""
        return self._condon_shortley

    @property
    def channel_convention(self):
        """the channel convention of the spherical harmonic coefficients."""
        return self._channel_convention

    @channel_convention.setter
    def channel_convention(self, value):
        """
        set the channel convention of the spherical harmonic coefficients.
        """
        if value not in ["acn", "fuma"]:
            raise ValueError("Invalid channel convention, has to be 'acn' "
                             f"or 'fuma', but is {value}")

        if self.channel_convention is not value:
            self._data = change_channel_convention(self._data,
                                                   self.channel_convention,
                                                   value, axis=-2)
            self._channel_convention = value
