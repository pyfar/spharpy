from pyfar import Signal, TimeData, FrequencyData
from pyfar.classes.audio import _Audio
from spharpy.spherical import renormalize, change_channel_convention
from spharpy.classes.sh import _SphericalHarmonicBase
import numpy as np
from abc import ABC


class _SphericalHarmonicAudio(_Audio, _SphericalHarmonicBase, ABC):
    """
    Base class for spherical harmonics audio objects.

    This class extends the pyfar Audio class with all methods and
    properties required for spherical harmonics data and are common to the
    three sub-classes :py:func:`SphericalHarmonicsTimeData`,
    :py:func:`SphericalHarmonicsFrequencyData`, and
    :py:func:`SphericalHarmonicsSignal`.

    Objects of this class contain spherical harmonics coefficients which are
    directly convertible between channel conventions ACN and FUMA, as
    well as the normalizations N3D, SN3D, or MaxN. The definition of
    the spherical harmonics basis functions is based on the scipy convention
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

        # check dimensions
        if len(self.cshape) < 2:
            raise ValueError("Invalid number of dimensions. Data should have "
                             "at least 3 dimensions.")

        # calculate n_max
        n_max = np.sqrt(self.cshape[-1])-1
        if n_max - int(n_max) != 0:
            raise ValueError("Invalid number of SH channels: "
                             f"{self._data.shape[-2]}. It must match "
                             "(n_max + 1)^2.")
        self._n_max = int(n_max)

        _SphericalHarmonicBase.__init__(
            self,
            basis_type,
            channel_convention,
            normalization,
            condon_shortley
        )

    @property
    def n_max(self):
        """Get or set the spherical harmonic order."""
        return int(np.sqrt(self._data.cshape[-1])-1)


class SphericalHarmonicTimeData(_SphericalHarmonicAudio, TimeData):
    """
    Create spherical harmonic audio object with time domain spherical
    harmonic coefficients and times.

    Objects of this class contain time data which is not directly convertible
    to the frequency domain, i.e., non-equidistant samples.

    Parameters
    ----------
    data : array, double
        Raw data in the time domain. The data should have at least 3
        dimensions, according to the 'C' memory layout, e.g. data of
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

        _SphericalHarmonicAudio.__init__(
            self, basis_type, normalization, channel_convention,
            condon_shortley)

        TimeData.__init__(self, data=data, times=times, comment=comment,
                          is_complex=is_complex)

    @property
    def time(self):
        """Return or set the time data."""
        data = TimeData.time.fget(self)

        # renormalize according to desired normalization
        if not self.normalization == "N3D":
            data = renormalize(data,
                               "ACN",
                               "N3D",
                               self.normalization,
                               axis=-2)
        # change channel convention according to desired convention
        if not self.channel_convention == "ACN":
            data = change_channel_convention(
                        data,
                        "ACN",
                        self.channel_convention,
                        axis=-2)
        return data

    @time.setter
    def time(self, value):
        """Return or set the time data."""
        # check dimensions
        if len(value.shape) < 3:
            raise ValueError("Invalid number of dimensions. Data should have "
                             "at least 3 dimensions.")

        # convert to N3D and ACN if necessary
        if not self.normalization == "N3D":
            value = renormalize(value,
                                self.channel_convention,
                                self.normalization,
                                "N3D",
                                axis=-2)
        if not self.channel_convention == "ACN":
            value = change_channel_convention(
                        value,
                        self.channel_convention,
                        "ACN",
                        axis=-2)
        TimeData.time.fset(self, value)


class SphericalHarmonicFrequencyData(_SphericalHarmonicAudio, Signal):
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
        3 dimensions, according to the 'C' memory layout, e.g. data of
        ``shape = (1, 4, 1024)`` has 1 channel with 4 spherical harmonic
        coefficients with 1024 frequency bins each.. Data can be ``int``,
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

        _SphericalHarmonicAudio.__init__(
            self, basis_type, normalization, channel_convention,
            condon_shortley)

        FrequencyData.__init__(self, data=data, frequencies=frequencies,
                               comment=comment)

    @property
    def freq(self):
        """Return or set the data in the frequency domain."""
        data = FrequencyData.freq.fget(self)

        # renormalize according to desired normalization
        if not self.normalization == "N3D":
            data = renormalize(data,
                               "ACN",
                               "N3D",
                               self.normalization,
                               axis=-2)
        # change channel convention according to desired convention
        if not self.channel_convention == "ACN":
            data = change_channel_convention(
                        data,
                        "ACN",
                        self.channel_convention,
                        axis=-2)
        return data

    @freq.setter
    def freq(self, value):
        """Return or set the data in the frequency domain."""
        if len(value.shape) < 3:
            raise ValueError("Invalid number of dimensions. Data should have "
                             "at least 3 dimensions.")

        # check dimensions
        if len(value.shape) < 3:
            raise ValueError("Invalid number of dimensions. Data should have "
                             "at least 3 dimensions.")

        # convert to N3D and ACN if necessary
        if not self.normalization == "N3D":
            value = renormalize(value,
                                self.channel_convention,
                                self.normalization,
                                "N3D",
                                axis=-2)
        if not self.channel_convention == "ACN":
            value = change_channel_convention(
                        value,
                        self.channel_convention,
                        "ACN",
                        axis=-2)

        FrequencyData.freq.fset(self, value)


class SphericalHarmonicSignal(_SphericalHarmonicAudio, Signal):
    """
    Create audio object with spherical harmonics coefficients in time or
    frequency domain.

    Objects of this class contain spherical harmonics coefficients which are
    directly convertible between time and frequency domain (equally spaced
    samples and frequency bins), the channel conventions ACN and FuMa, as
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

        _SphericalHarmonicAudio.__init__(
            self, basis_type, normalization, channel_convention,
            condon_shortley)

        Signal.__init__(self, data=data, sampling_rate=sampling_rate,
                        n_samples=n_samples, domain=domain, fft_norm=fft_norm,
                        comment=comment, is_complex=is_complex)

        @property
        def time(self):
            """Return or set the data in the time domain."""
            data = Signal.time.fget(self)

            # renormalize according to desired normalization
            if not self.normalization == "N3D":
                data = renormalize(data,
                                   "ACN",
                                   "N3D",
                                   self.normalization,
                                   axis=-2)
            # change channel convention according to desired convention
            if not self.channel_convention == "ACN":
                data = change_channel_convention(
                            data,
                            "ACN",
                            self.channel_convention,
                            axis=-2)
            return data

        @time.setter
        def time(self, value):
            """Return or set the data in the time domain."""
            if len(value.shape) < 3:
                raise ValueError("Invalid number of dimensions. Data should have "
                                 "at least 3 dimensions.")

            # check dimensions
            if len(value.shape) < 3:
                raise ValueError("Invalid number of dimensions. Data should have "
                                 "at least 3 dimensions.")

            # convert to N3D and ACN if necessary
            if not self.normalization == "N3D":
                value = renormalize(value,
                                    self.channel_convention,
                                    self.normalization,
                                    "N3D",
                                    axis=-2)
            if not self.channel_convention == "ACN":
                value = change_channel_convention(
                            value,
                            self.channel_convention,
                            "ACN",
                            axis=-2)

            Signal.time.fset(self, value)

        @FrequencyData.freq.getter
        def freq(self):
            """Return or set the normalized frequency domain data."""
            data = Signal.freq.fget(self)

            # renormalize according to desired normalization
            if not self.normalization == "N3D":
                data = renormalize(data,
                                   "ACN",
                                   "N3D",
                                   self.normalization,
                                   axis=-2)
            # change channel convention according to desired convention
            if not self.channel_convention == "ACN":
                data = change_channel_convention(
                            data,
                            "ACN",
                            self.channel_convention,
                            axis=-2)
            return data

        @freq.setter
        def freq(self, value):
            """Return or set the data in the frequency domain."""
            if len(value.shape) < 3:
                raise ValueError("Invalid number of dimensions. Data should have "
                                 "at least 3 dimensions.")

            # check dimensions
            if len(value.shape) < 3:
                raise ValueError("Invalid number of dimensions. Data should have "
                                 "at least 3 dimensions.")

            # convert to N3D and ACN if necessary
            if not self.normalization == "N3D":
                value = renormalize(value,
                                    self.channel_convention,
                                    self.normalization,
                                    "N3D",
                                    axis=-2)
            if not self.channel_convention == "ACN":
                value = change_channel_convention(
                            value,
                            self.channel_convention,
                            "ACN",
                            axis=-2)

            Signal.freq.fset(self, value)
