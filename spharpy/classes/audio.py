"""
The spherical harmonic (SH) audio classes store audio data in the SH domain.
Please refer to the
:doc:`/theory/spherical_harmonic_definition` page for more general information.

The spherical harmonic audio classes are build upon the
:py:mod:`pyfar audio classes<pyfar.classes.audio>` and we recommend to get
familiar with these classes before continuing.

In addition to all functionality provided by the pyfar audio classes, the
spherical harmonic audio classes allow to store parameters defining the
spherical harmonics, these are the ``basis_type``, ``normalization``,
``channel_convention`` and the ``condon_shortley`` phase convention.
The last dimension of the channel shape must always match a valid number of
spherical harmonics, i.e. :math:`(N+1)^2`, where  :math:`N` is the spherical
harmonic order for which the audio data is created. The spherical harmonic
order of the data contained in the signal can be accessed through the property
``n_max``.

A SH signal can be created either directly

>>> import spharpy
>>> data = [[0, 0],  # data of first SH channel
...         [1, 1],  # data of second SH channel
...         [2, 2],  # data of third SH channel
...         [3, 4]]  # data of fourth SH channel
>>> sh_signal = spharpy.SphericalHarmonicSignal(
...     data, 44100, basis_type='real', normalization='N3D',
...     channel_convention='ACN', condon_shortley=False)

or from an SH definition

>>> # create a SH definition with default parameters
>>> definition = spharpy.SphericalHarmonicDefinition()
>>> sh_signal = spharpy.SphericalHarmonicSignal.from_definition(
...     definition, data, 44100)

Both examples create a first order SH signal with four SH channels and two time
samples at a sampling rate of 44.1 kHz.
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

    data = np.atleast_2d(data)
    return data[np.newaxis, ...] if data.ndim < 3 else data


def _assert_valid_number_of_sh_channels(shape, sh_axis):
    """Check if the channel shape matches an integer spherical harmonic order.

    Parameters
    ----------
    shape : tuple, int
        Shape of the data array.
    sh_axis : int
        Specifies which axis of data holds the spherical harmonic coefficients.
        Negative indexing, i.e., interpreted relative to the end of the array.

    Raises
    ------
    ValueError
        Raised if the number of spherical harmonic channels does not match
        (n_max + 1)^2 for an integer n_max.
    """

    sh_channels = shape[sh_axis]
    n_max = np.sqrt(sh_channels)-1
    if n_max - int(n_max) != 0:
        raise ValueError(
            "Invalid number of spherical harmonic channels: "
            f"{sh_channels}. It must match (n_max + 1)^2.")


def _convert_to_standard_definition(
        data,
        normalization,
        channel_convention,
        sh_axis=-2):
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
    sh_axis : int
        Specifies which axis of data holds the spherical harmonic coefficients.
        Negative indexing, i.e., interpreted relative to the end of the array.

    Returns
    -------
    numpy.ndarray
        Spherical harmonic data following the standard definition (N3D, ACN).
    """

    data = renormalize(
        data, channel_convention, normalization,
        "N3D", axis=sh_axis)

    data = change_channel_convention(
        data, channel_convention, "ACN", axis=sh_axis)

    return data


def _convert_from_standard_definition(
        data,
        normalization,
        channel_convention,
        sh_axis=-2):
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
    sh_axis : int
        Specifies which axis of data holds the spherical harmonic coefficients.
        Negative indexing, i.e., interpreted relative to the end of the array.

    Returns
    -------
    numpy.ndarray
        Spherical harmonic data according to the desired definition.
    """

    data = renormalize(
        data, "ACN", "N3D", normalization, axis=sh_axis)

    data = change_channel_convention(
        data, "ACN", channel_convention, axis=sh_axis)

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
        Normalization convention, either ``'N3D'``, ``'NM'``, ``'SN3D'``,
        ``'SNM'``, or ``'maxN'``. ``'maxN'`` is only supported up to 3rd order.
    channel_convention : str
        Channel ordering convention, either ``'ACN'`` or ``'FuMa'``.
        ``'FuMa'`` is only supported up to 3rd order.
    condon_shortley : bool
        Whether to include the Condon-Shortley phase term. If ``True``,
        Condon-Shortley is included, if ``False`` it is not
        included. ``'auto'`` corresponds to ``True`` for complex `basis_type`
        and ``False`` for real `basis_type`.
    domain : ``'time'``, ``'freq'``, optional
        Domain of data. The default is ``'time'``
    comment : str
        A comment related to `data`. The default is ``None``.
    sh_caxis : int
        Specifies which axis of data holds the spherical harmonic coefficients.
        Negative indexing, i.e., interpreted relative to the end of the array.

    """

    def __init__(self, basis_type, normalization, channel_convention,
                 condon_shortley, sh_caxis):

        _SphericalHarmonicBase.__init__(
            self,
            basis_type,
            normalization,
            channel_convention,
            condon_shortley)

        self._sh_caxis = sh_caxis

    @property
    def n_max(self):
        """Get or set the spherical harmonic order."""
        return int(np.sqrt(self.cshape[-1])-1)

    @property
    def sh_caxis(self):
        """Get the spherical harmonic axis"""
        return self._sh_caxis

    @_SphericalHarmonicBase.basis_type.setter
    def basis_type(self, value):
        """Get or set the spherical harmonic basis type."""

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
    condon_shortley : bool or str
        Whether to include the Condon-Shortley phase term. If ``True``,
        Condon-Shortley is included, if ``False`` it is not
        included. ``'auto'`` corresponds to ``True`` for complex `basis_type`
        and ``False`` for real `basis_type`.
    comment : str
        A comment related to `data`. The default is ``""``.
    is_complex : bool, optional
        A flag which indicates if the time data are real or complex-valued.
        The default is ``False``.
    sh_caxis : int
        Specifies which axis of data holds the spherical harmonic coefficients.
        Negative indexing, i.e., interpreted relative to the end of the array.
        default is second last axis (-2).
    """

    def __init__(self, data, times, basis_type, normalization,
                 channel_convention, condon_shortley, comment="",
                 is_complex=False, sh_caxis=-2):

        if not is_complex and basis_type == 'complex':
            raise ValueError(
                "Complex spherical harmonic basis requires "
                "complex time data. Set is_complex=True.")

        data = _atleast_3d_first_dimension(data)
        _assert_valid_number_of_sh_channels(data.shape, sh_caxis)

        _SphericalHarmonicAudio.__init__(
            self, basis_type, normalization, channel_convention,
            condon_shortley, sh_caxis=sh_caxis)

        TimeData.__init__(self, data=data, times=times, comment=comment,
                          is_complex=is_complex)

    @classmethod
    def from_definition(
            cls, sh_definition, data, times, comment="", is_complex=False,
            sh_caxis=-2):
        r"""
        Create a SphericalHarmonicTimeData class object from
        SphericalHarmonicDefinition object, data, and times.

        Parameters
        ----------
        sh_definition : SphericalHarmonicDefinition
            The spherical harmonic definition.
        data : array, double
            Raw data in the time domain. The data should have at least 2
            dimensions, with the last dimension representing the time domain
            samples, the second to last the spherical harmonic coefficients,
            and any leading dimensions representing optional channels.
            Accordingly, the data should follow the 'C' memory layout, e.g.
            data of ``shape = (1, 4, 1024)`` has 1 channel with 4 spherical
            harmonic coefficients with 1024 samples each. The data can be
            ``int``, ``float`` or ``complex``. Data of type ``int`` is
            converted to ``float``.
        times : array, double
            Times in seconds at which the data is sampled. The number of times
            must match the size of the last dimension of `data`, i.e.,
            ``data.shape[-1]``.
        comment : str
            A comment related to `data`. The default is ``None``.
        is_complex : bool, optional
            A flag which indicates if the time data are real or complex-valued.
            The default is ``False``.
        sh_caxis : int
            Specifies which axis of data holds the spherical harmonic
            coefficients. Negative indexing, i.e., interpreted relative to the
            end of the array. The default is second last axis (-2).
        """
        return cls(data, times,
                   basis_type=sh_definition.basis_type,
                   normalization=sh_definition.normalization,
                   channel_convention=sh_definition.channel_convention,
                   condon_shortley=sh_definition.condon_shortley,
                   comment=comment, is_complex=is_complex, sh_caxis=sh_caxis)

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
        _assert_valid_number_of_sh_channels(value.shape, self._sh_caxis)

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
        Normalization convention, either ``'N3D'``, ``'NM'``, ``'SN3D'``,
        ``'SNM'``, or ``'maxN'``. ``'maxN'`` is only supported up to 3rd order.
    channel_convention : str
        Channel ordering convention, either ``'ACN'`` or ``'FuMa'``.
        ``'FuMa'`` is only supported up to 3rd order.
    condon_shortley : bool or str
        Whether to include the Condon-Shortley phase term. If ``True``,
        Condon-Shortley is included, if ``False`` it is not
        included. ``'auto'`` corresponds to ``True`` for complex `basis_type`
        and ``False`` for real `basis_type`.
    comment : str
        A comment related to `data`. The default is ``""``.
    sh_caxis : int
        Specifies which axis of data holds the spherical harmonic coefficients.
        Negative indexing, i.e., interpreted relative to the end of the array.
        The default is second last axis (-2).
    """

    def __init__(self, data, frequencies, basis_type, normalization,
                 channel_convention, condon_shortley, comment="", sh_caxis=-2):

        data = _atleast_3d_first_dimension(data)
        _assert_valid_number_of_sh_channels(data.shape, sh_caxis)

        _SphericalHarmonicAudio.__init__(
            self, basis_type, normalization, channel_convention,
            condon_shortley, sh_caxis=sh_caxis)

        FrequencyData.__init__(self, data=data, frequencies=frequencies,
                               comment=comment)

    @classmethod
    def from_definition(
            cls, sh_definition, data, frequencies, comment="", sh_caxis=-2):
        r"""
        Create a SphericalHarmonicFrequencyData class object from
        SphericalHarmonicDefinition object, data, and frequencies
        rate.

        Parameters
        ----------
        sh_definition : SphericalHarmonicDefinition
            The spherical harmonic definition.
        data : ndarray, double
            Raw data in the frequency domain. The data should have at least
            2 dimensions, with the last dimension representing the frequency
            domain bins, the second to last the spherical harmonic
            coefficients, and any leading dimensions representing optional
            channels. Accordingly, the data should follow the 'C' memory
            layout, e.g. data of ``shape = (1, 4, 1024)`` has 1 channel with 4
            spherical harmonic coefficients with 1024 frequency bins each. The
            data can be ``int``, ``float`` or ``complex``. Data of type
            ``int`` is converted to ``float``.
        frequencies : array, double
            Frequencies of the data in Hz. The number of frequencies must match
            the size of the last dimension of `data`, i.e., ``data.shape[-1]``.
        comment : str
            A comment related to `data`. The default is ``None``.
        sh_caxis : int
            Specifies which axis of data holds the spherical harmonic
            coefficients. Negative indexing, i.e., interpreted relative to the
            end of the array. The default is second last axis (-2).
        """
        return cls(data, frequencies,
                   basis_type=sh_definition.basis_type,
                   normalization=sh_definition.normalization,
                   channel_convention=sh_definition.channel_convention,
                   condon_shortley=sh_definition.condon_shortley,
                   comment=comment, sh_caxis=sh_caxis)

    @property
    def freq(self):
        """Return or set the data in the frequency domain."""
        return _convert_from_standard_definition(FrequencyData.freq.fget(self),
                                                 self.normalization,
                                                 self.channel_convention,
                                                 self._sh_caxis)

    @freq.setter
    def freq(self, value):
        """Return or set the data in the frequency domain."""
        value = _atleast_3d_first_dimension(value)
        _assert_valid_number_of_sh_channels(value.shape, self._sh_caxis)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention)

        FrequencyData.freq.fset(self, value)


class SphericalHarmonicSignal(_SphericalHarmonicAudio, Signal):
    """
    Create audio object with spherical harmonic coefficients in time or
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
        Normalization convention, either ``'N3D'``, ``'NM'``, ``'SN3D'``,
        ``'SNM'``, or ``'maxN'``. ``'maxN'`` is only supported up to 3rd order.
    channel_convention : str
        Channel ordering convention, either ``'ACN'`` or ``'FuMa'``.
        ``'FuMa'`` is only supported up to 3rd order.
    condon_shortley : bool or str
        Whether to include the Condon-Shortley phase term. If ``True``,
        Condon-Shortley is included, if ``False`` it is not
        included. ``'auto'`` corresponds to ``True`` for complex `basis_type`
        and ``False`` for real `basis_type`.
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
    sh_caxis : int
        Specifies which axis of data holds the spherical harmonic coefficients.
        Negative indexing, i.e., interpreted relative to the end of the array.
        The default is second last axis (-2).

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
                 is_complex=False,
                 sh_caxis=-2):

        data = _atleast_3d_first_dimension(data)
        _assert_valid_number_of_sh_channels(data.shape, sh_caxis)

        _SphericalHarmonicAudio.__init__(
            self, basis_type, normalization, channel_convention,
            condon_shortley, sh_caxis=sh_caxis)

        Signal.__init__(self, data=data, sampling_rate=sampling_rate,
                        n_samples=n_samples, domain=domain, fft_norm=fft_norm,
                        comment=comment, is_complex=is_complex)

    @classmethod
    def from_definition(
            cls, sh_definition, data, sampling_rate, domain='time',
            fft_norm='none', comment="", is_complex=False, sh_caxis=-2):
        r"""
        Create a SphericalHarmonicSignal class object from
        SphericalHarmonicDefinition object, data, and sampling
        rate.

        Parameters
        ----------
        sh_definition : SphericalHarmonicDefinition
            The spherical harmonic definition.
        data : ndarray, double
            Raw data of the spherical harmonic signal in the time or
            frequency domain. The data should have at least 2 dimensions, with
            the last dimension representing the time domain
            samples/frequency domain bins, the second to last the spherical
            harmonic coefficients, and any leading dimensions representing
            optional channels. Accordingly, the data should follow the 'C'
            memory layout, e.g. data of ``shape = (1, 4, 1024)`` has 1 channel
            with 4 spherical harmonic coefficients with 1024 samples or
            frequency bins each. Time data is converted to ``float``.
            Frequency is converted to ``complex`` and must be provided as
            single sided spectra, i.e., for all frequencies between 0 Hz and
            half the sampling rate.
        sampling_rate : double
            Sampling rate in Hz
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
        sh_caxis : int
            Specifies which axis of data holds the spherical harmonic
            coefficients. Negative indexing, i.e., interpreted relative to the
            end of the array. The default is second last axis (-2).
        """
        return cls(data, sampling_rate,
                   basis_type=sh_definition.basis_type,
                   normalization=sh_definition.normalization,
                   channel_convention=sh_definition.channel_convention,
                   condon_shortley=sh_definition.condon_shortley,
                   domain=domain, fft_norm=fft_norm,
                   comment=comment, is_complex=is_complex, sh_caxis=sh_caxis)

    @property
    def freq(self):
        """Return or set the data in the frequency domain."""
        return _convert_from_standard_definition(Signal.freq.fget(self),
                                                 self.normalization,
                                                 self.channel_convention,
                                                 self._sh_caxis)

    @freq.setter
    def freq(self, value):
        """Return or set the data in the frequency domain."""
        value = _atleast_3d_first_dimension(value)
        _assert_valid_number_of_sh_channels(value.shape, self._sh_caxis)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention)

        Signal.freq.fset(self, value)

    @property
    def freq_raw(self):
        """Return or set the frequency domain data without normalization."""
        return _convert_from_standard_definition(Signal.freq_raw.fget(self),
                                                 self.normalization,
                                                 self.channel_convention,
                                                 self._sh_caxis)

    @freq_raw.setter
    def freq_raw(self, value):
        """Return or set the frequency domain data without normalization."""
        value = _atleast_3d_first_dimension(value)
        _assert_valid_number_of_sh_channels(value.shape, self._sh_caxis)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention, self._sh_caxis)

        Signal.freq_raw.fset(self, value)

    @property
    def time(self):
        """Return or set the time data."""
        return _convert_from_standard_definition(Signal.time.fget(self),
                                                 self.normalization,
                                                 self.channel_convention,
                                                 self._sh_caxis)

    @time.setter
    def time(self, value):
        """Return or set the time data."""
        value = _atleast_3d_first_dimension(value)
        _assert_valid_number_of_sh_channels(value.shape, self._sh_caxis)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention, self._sh_caxis)
        Signal.time.fset(self, value)
