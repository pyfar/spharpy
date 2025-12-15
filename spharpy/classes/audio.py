"""
Documentation for all SphericalHarmonics Audio classes will be provided in
another PR.
"""
from pyfar import Signal, TimeData, FrequencyData
from pyfar.classes.audio import _Audio
from spharpy.spherical import renormalize, change_channel_convention
from spharpy.classes.sh import _SphericalHarmonicBase
import numpy as np
from abc import ABC


def _atleast_3d_first_dimension(data):
    """Ensure that data has at least 3 dimensions.
    Adds a singleton dimensions at the front if necessary.
    """

    return np.atleast_2d(data)[np.newaxis, ...] if data.ndim < 3 else data


def _assert_valid_number_of_sh_channels(shape):
    """Check if the channel shape matches an integer spherical harmonic order.

    Parameters
    ----------
    shape : tuple, int
        Shape of the data array.
    Raises
    ------
    ValueError
        Raised if the number of spherical harmonic channels does not match
        (n_max + 1)^2 for an integer n_max.
    """

    sh_channels = shape[-2]
    n_max = np.sqrt(sh_channels)-1
    if n_max - int(n_max) != 0:
        raise ValueError(
            "Invalid number of spherical harmonic channels: "
            f"{sh_channels}. It must match (n_max + 1)^2.")


def _convert_to_standard_definition(
        data,
        normalization,
        channel_convention):
    """Convert data to the standard spherical harmonic definition.

    Parameters
    ----------
    data : numpy.ndarray
        Spherical harmonic data.
    normalization : str
        Normalization convention, either ``'N3D'``, ``'NM'``,
        ``'maxN'``, ``'SN3D'`` or ``'SNM'``. (maxN is only supported up
        to 3rd order)
    channel_convention : str
        Channel ordering convention, either ``'ACN'`` or ``'FuMa'``.
        (FuMa is only supported up to 3rd order)

    Returns
    -------
    numpy.ndarray
        Spherical harmonic data following the standard definition (N3D, ACN).
    """

    data = renormalize(
        data, channel_convention, normalization,
        "N3D", axis=-2)

    data = change_channel_convention(
        data, channel_convention, "ACN", axis=-2)

    return data


def _convert_from_standard_definition(
        data,
        normalization,
        channel_convention):
    """Convert data from standard definition to the desired one.

    Parameters
    ----------
    data : numpy.ndarray
        Spherical harmonic data using the standard definition (N3D, ACN).
    normalization : str
        Normalization convention, either ``'N3D'``, ``'NM'``,
        ``'maxN'``, ``'SN3D'`` or ``'SNM'``. (maxN is only supported up
        to 3rd order)
    channel_convention : str
        Channel ordering convention, either ``'ACN'`` or ``'FuMa'``.
        (FuMa is only supported up to 3rd order)

    Returns
    -------
    numpy.ndarray
        Spherical harmonic data according to the desired definition.
    """

    data = renormalize(
        data, "ACN", "N3D", normalization, axis=-2)

    data = change_channel_convention(
        data, "ACN", channel_convention, axis=-2)

    return data


class _SphericalHarmonicAudio(_Audio, _SphericalHarmonicBase, ABC):
    """
    Base class for spherical harmonics audio objects.

    This class extends the pyfar Audio class with all methods and
    properties required for spherical harmonic data and are common to the
    three sub-classes :py:class:`SphericalHarmonicTimeData`,
    :py:class:`SphericalHarmonicFrequencyData`, and
    :py:class:`SphericalHarmonicSignal`.

    Objects of this class contain spherical harmonic coefficients which are
    directly convertible between channel conventions ACN and FUMA, as
    well as the normalizations N3D, SN3D, or MaxN. The definition of
    the spherical harmonic basis functions is based on the scipy convention
    which includes the Condon-Shortley phase.


    Parameters
    ----------
    basis_type : str
        Type of spherical harmonic basis, either ``'complex'`` or
        ``'real'``.
    normalization : str
        Normalization convention, either ``'N3D'``, ``'NM'``,
        ``'maxN'``, ``'SN3D'`` or ``'SNM'``. (maxN is only supported up
        to 3rd order)
    channel_convention : str
        Channel ordering convention, either ``'ACN'`` or ``'FuMa'``.
        (FuMa is only supported up to 3rd order)
    condon_shortley : bool
        Flag to indicate if the Condon-Shortley phase term is included
        (``True``) or not (``False``).
    domain : ``'time'``, ``'freq'``, optional
        Domain of data. The default is ``'time'``
    comment : str
        A comment related to `data`. The default is ``None``.

    """

    def __init__(self, basis_type, normalization, channel_convention,
                 condon_shortley):

        _SphericalHarmonicBase.__init__(
            self,
            basis_type,
            channel_convention,
            normalization,
            condon_shortley)

    @property
    def n_max(self):
        """Get or set the spherical harmonic order."""
        return int(np.sqrt(self.cshape[-1])-1)

    @_SphericalHarmonicBase.basis_type.setter
    def basis_type(self, value):
        """Get or set the type of spherical harmonic basis."""

        # Make sure that the basis type can only be set during initialization.
        # Changing it afterwards requires implementing the conversion between
        # real and complex-valued coefficients, which is possible but not yet
        # implemented.
        if self._basis_type is not None:
            raise AttributeError("Changing the basis_type is not yet "
                                 "supported.")
        else:
            _SphericalHarmonicBase.basis_type.fset(self, value)


class SphericalHarmonicTimeData(_SphericalHarmonicAudio, TimeData):
    """
    Create spherical harmonic audio object with time domain spherical
    harmonic coefficients and times.

    Objects of this class contain time data which is not directly convertible
    to the frequency domain, i.e., non-equidistant temporal sampling.

    Parameters
    ----------
    data : array, double
        Raw data in the time domain. The data should have at least 2
        dimensions, with the last dimension representing the time domain
        samples, the second to last the spherical harmonic coefficients,
        and any leading dimensions representing optional channels. Accordingly,
        the data should follow the 'C' memory layout, e.g. data of
        ``shape = (1, 4, 1024)`` has 1 channel with 4 spherical harmonic
        coefficients with 1024 samples each. The data can be ``int``,
        ``float`` or ``complex``. Data of type ``int`` is converted to
        ``float``.
    times : array, double
        Times in seconds at which the data is sampled. The number of times
        must match the size of the last dimension of `data`, i.e.,
        ``data.shape[-1]``.
    basis_type : str
        Type of spherical harmonic basis, either ``'complex'`` or
        ``'real'``.
    normalization : str
        Normalization convention, either ``'N3D'``, ``'NM'``,
        ``'maxN'``, ``'SN3D'`` or ``'SNM'``. (maxN is only supported up
        to 3rd order)
    channel_convention : str
        Channel ordering convention, either ``'ACN'`` or ``'FuMa'``.
        (FuMa is only supported up to 3rd order)
    condon_shortley : bool
        Flag to indicate if the Condon-Shortley phase term is included
        (``True``) or not (``False``).
    comment : str
        A comment related to `data`. The default is ``""``.
    is_complex : bool, optional
        A flag which indicates if the time data are real or complex-valued.
        The default is ``False``.
    """

    def __init__(self, data, times, basis_type, normalization,
                 channel_convention, condon_shortley, comment="",
                 is_complex=False):

        if not is_complex and basis_type == 'complex':
            raise ValueError(
                "Complex spherical harmonic basis requires "
                "complex time data. Set is_complex=True.")

        data = _atleast_3d_first_dimension(data)
        _assert_valid_number_of_sh_channels(data.shape)

        _SphericalHarmonicAudio.__init__(
            self, basis_type, normalization, channel_convention,
            condon_shortley)

        TimeData.__init__(self, data=data, times=times, comment=comment,
                          is_complex=is_complex)

    @property
    def time(self):
        """Return or set the time data."""
        return _convert_from_standard_definition(TimeData.time.fget(self),
                                                 self.normalization,
                                                 self.channel_convention)

    @time.setter
    def time(self, value):
        """Return or set the time data."""
        value = _atleast_3d_first_dimension(value)
        _assert_valid_number_of_sh_channels(value.shape)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention)
        TimeData.time.fset(self, value)


class SphericalHarmonicFrequencyData(_SphericalHarmonicAudio, FrequencyData):
    """
    Create spherical harmonic audio object with frequency domain spherical
    harmonic coefficients and frequencies.

    Objects of this class contain frequency data which is not directly
    convertible to the time domain, i.e., non-equidistantly spaced bins or
    incomplete spectra.

    Parameters
    ----------
    data : array, double
        Raw data in the frequency domain. The data should have at least
        2 dimensions, with the last dimension representing the frequency domain
        bins, the second to last the spherical harmonic coefficients,
        and any leading dimensions representing optional channels. Accordingly,
        the data should follow the 'C' memory layout, e.g. data of
        ``shape = (1, 4, 1024)`` has 1 channel with 4 spherical harmonic
        coefficients with 1024 frequency bins each. The data can be ``int``,
        ``float`` or ``complex``. Data of type ``int`` is converted to
        ``float``.
    frequencies : array, double
        Frequencies of the data in Hz. The number of frequencies must match
        the size of the last dimension of `data`, i.e., ``data.shape[-1]``.
    basis_type : str
        Type of spherical harmonic basis, either ``'complex'`` or
        ``'real'``.
    normalization : str
        Normalization convention, either ``'N3D'``, ``'NM'``,
        ``'maxN'``, ``'SN3D'`` or ``'SNM'``. (maxN is only supported up
        to 3rd order)
    channel_convention : str
        Channel ordering convention, either ``'ACN'`` or ``'FuMa'``.
        (FuMa is only supported up to 3rd order)
    condon_shortley : bool
        Flag to indicate if the Condon-Shortley phase term is included
        (``True``) or not (``False``).
    comment : str
        A comment related to `data`. The default is ``""``.
    """

    def __init__(self, data, frequencies, basis_type, normalization,
                 channel_convention, condon_shortley, comment=""):

        data = _atleast_3d_first_dimension(data)
        _assert_valid_number_of_sh_channels(data.shape)

        _SphericalHarmonicAudio.__init__(
            self, basis_type, normalization, channel_convention,
            condon_shortley)

        FrequencyData.__init__(self, data=data, frequencies=frequencies,
                               comment=comment)

    @property
    def freq(self):
        """Return or set the data in the frequency domain."""
        return _convert_from_standard_definition(FrequencyData.freq.fget(self),
                                                 self.normalization,
                                                 self.channel_convention)

    @freq.setter
    def freq(self, value):
        """Return or set the data in the frequency domain."""
        value = _atleast_3d_first_dimension(value)
        _assert_valid_number_of_sh_channels(value.shape)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention)

        FrequencyData.freq.fset(self, value)


class SphericalHarmonicSignal(_SphericalHarmonicAudio, Signal):
    """
    Create audio object with spherical harmonics coefficients in time or
    frequency domain.

    Objects of this class contain spherical harmonic coefficients which are
    directly convertible between time and frequency domain (equally spaced
    samples and frequency bins), the channel conventions ACN and FuMa, as
    well as the normalizations N3D, SN3D, or MaxN, see [#]_. The definition of
    the spherical harmonics basis functions is based on the scipy convention
    which includes the Condon-Shortley phase, [#]_, [#]_.


    Parameters
    ----------
    data : ndarray, double
        Raw data of the spherical harmonics signal in the time or
        frequency domain. The data should have at least 2 dimensions, with
        the last dimension representing the time domain
        samples/frequency domain bins, the second to last the spherical
        harmonic coefficients, and any leading dimensions representing
        optional channels. Accordingly, the data should follow the 'C'
        memory layout, e.g. data of ``shape = (1, 4, 1024)`` has 1 channel
        with 4 spherical harmonic coefficients with 1024 samples or frequency
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
        Normalization convention, either ``'N3D'``, ``'NM'``,
        ``'maxN'``, ``'SN3D'`` or ``'SNM'``. (maxN is only supported up
        to 3rd order)
    channel_convention : str
        Channel ordering convention, either ``'ACN'`` or ``'FuMa'``.
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
        or ``'psd'``. See :py:func:`~pyfar.dsp.fft.normalization`
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

    def __init__(self,
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

        data = _atleast_3d_first_dimension(data)
        _assert_valid_number_of_sh_channels(data.shape)

        _SphericalHarmonicAudio.__init__(
            self, basis_type, normalization, channel_convention,
            condon_shortley)

        Signal.__init__(self, data=data, sampling_rate=sampling_rate,
                        n_samples=n_samples, domain=domain, fft_norm=fft_norm,
                        comment=comment, is_complex=is_complex)

    @classmethod
    def from_spherical_harmonics_definition(
            cls, data, sampling_rate, sh_definition, domain='time',
            fft_norm='none', comment="", is_complex=False):
        r"""
        Create a SphericalHarmonicsSignal class object from
        SphericalHarmonicsDefinition object, data, and sampling
        rate.
        """
        return cls(data, sampling_rate,
                   basis_type=sh_definition.basis_type,
                   normalization=sh_definition.normalization,
                   channel_convention=sh_definition.channel_convention,
                   condon_shortley=sh_definition.condon_shortley,
                   domain=domain, fft_norm=fft_norm,
                   comment=comment, is_complex=is_complex)

    @property
    def freq(self):
        """Return or set the data in the frequency domain."""
        return _convert_from_standard_definition(Signal.freq.fget(self),
                                                 self.normalization,
                                                 self.channel_convention)

    @freq.setter
    def freq(self, value):
        """Return or set the data in the frequency domain."""
    @property
    def freq(self):
        """Return or set the data in the frequency domain."""
        return _convert_from_standard_definition(Signal.freq.fget(self),
                                                 self.normalization,
                                                 self.channel_convention)

    @freq.setter
    def freq(self, value):
        """Return or set the data in the frequency domain."""
        value = _atleast_3d_first_dimension(value)
        _assert_valid_number_of_sh_channels(value.shape)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention)

        Signal.freq.fset(self, value)

    @property
    def freq_raw(self):
        """Return or set the frequency domain data without normalization."""
        return _convert_from_standard_definition(Signal.freq_raw.fget(self),
                                                 self.normalization,
                                                 self.channel_convention)

    @freq_raw.setter
    def freq_raw(self, value):
        """Return or set the frequency domain data without normalization."""
        value = _atleast_3d_first_dimension(value)
        _assert_valid_number_of_sh_channels(value.shape)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention)

        Signal.freq_raw.fset(self, value)

    @property
    def time(self):
        """Return or set the time data."""
        return _convert_from_standard_definition(Signal.time.fget(self),
                                                 self.normalization,
                                                 self.channel_convention)

    @time.setter
    def time(self, value):
        """Return or set the time data."""
        value = _atleast_3d_first_dimension(value)
        _assert_valid_number_of_sh_channels(value.shape)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention)
        Signal.time.fset(self, value)
