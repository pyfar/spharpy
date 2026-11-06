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


def _assert_valid_caxis_spherical_harmonics(ndim, caxis_spherical_harmonics):
    """Check if the spherical harmonic channel axes are ``int`` and do
       not exceed the number of dimensions of the data.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the data.
    caxis_spherical_harmonics : tuple
        Specifies which axis of data holds the spherical harmonic coefficients.
        Negative indexing, i.e., interpreted relative to the end of the array.

    Raises
    ------
    ValueError
        Raised if caxis_spherical_harmonics does not only contain integers, or
        does contain values greater than data's number of dimensions.
    """

    for caxis in caxis_spherical_harmonics:
        if not isinstance(caxis, int):
            raise ValueError(
                "caxis_spherical_harmonics must contain only integers, "
                f"but got {type(caxis).__name__}.")

        if abs(caxis) > ndim:
            raise ValueError(
                "caxis_spherical_harmonics contains invalid axis. "
                f"Axis must be in the range [{-ndim}, {ndim}], "
                f"but is {caxis}.")


def _assert_valid_number_of_sh_channels(shape, caxis_spherical_harmonics):
    """Check if the channel shape matches an integer spherical harmonic order.

    Parameters
    ----------
    shape : tuple, int
        Shape of the data array.
    caxis_spherical_harmonics : tuple
        Specifies which axis of data holds the spherical harmonic coefficients.
        Negative indexing, i.e., interpreted relative to the end of the array.

    Raises
    ------
    ValueError
        Raised if the number of spherical harmonic channels does not match
        (n_max + 1)^2 for an integer n_max.
    """
    # convert channel axes to axes in data
    axes = tuple(x - 1 for x in caxis_spherical_harmonics)

    # check if all spherical harmonic channels have same length
    sh_channels = [shape[a] for a in axes]
    if len(set(sh_channels)) != 1:
        raise ValueError("All SH axes must have the same number of channels, "
                         f"but got {sh_channels}.")

    # check if all spherical harmonic channels match n_max
    sh_channels = sh_channels[0]
    n_max = np.sqrt(sh_channels)-1
    if n_max - int(n_max) != 0:
        raise ValueError(
            "Invalid number of spherical harmonic channels: "
            f"{sh_channels}. It must match (n_max + 1)^2.")


def _init_spherical_harmonic_channels(cshape, caxis_spherical_harmonics):
    """Create the arrays tracking the spherical harmonic channels.

    Parameters
    ----------
    cshape : tuple
        Channel shape of the data.
    caxis_spherical_harmonics : tuple
        Specifies which channel axis (or axes) holds the spherical harmonic
        coefficients. Negative indexing, i.e., interpreted relative to the
        end of `cshape`.

    Returns
    -------
    is_sh_channel : numpy.ndarray
        Boolean array of shape `cshape` that is ``True`` for each channel
        containing spherical harmonic coefficients.
    acn : numpy.ndarray
        Integer array of shape ``cshape + (len(caxis_spherical_harmonics), )``
        containing the ambisonic channel number (ACN) of each channel. The
        last dimension holds the ACN with respect to each spherical harmonic
        channel axis.
    """
    cdim = len(cshape)
    is_sh_channel = np.ones(cshape, dtype=bool)
    acn = np.zeros(cshape + (len(caxis_spherical_harmonics), ), dtype=int)

    for idx, caxis in enumerate(caxis_spherical_harmonics):
        # ACNs increase along the spherical harmonic axis and are constant
        # along all other axes
        shape = [1] * cdim
        shape[caxis] = cshape[caxis]
        acn[..., idx] = np.arange(cshape[caxis]).reshape(shape)

    return is_sh_channel, acn


def _is_spherical_harmonic_axis(acn, axis):
    """Check if the ambisonic channel numbers (ACN) are valid along an axis.

    The ACNs are valid along `axis` if they are ``0, 1, 2, ..., (N+1)^2-1``
    along `axis` and constant along all other axes.

    Parameters
    ----------
    acn : numpy.ndarray
        Integer array of ambisonic channel numbers of shape `cshape`.
    axis : int
        Channel axis to check.

    Returns
    -------
    is_sh_axis : bool
        ``True`` if `axis` holds valid spherical harmonic channels.
    """
    n_channels = acn.shape[axis]

    # number of channels must match (N+1)^2
    if not np.sqrt(n_channels).is_integer():
        return False

    shape = [1] * acn.ndim
    shape[axis] = n_channels
    expected = np.broadcast_to(
        np.arange(n_channels).reshape(shape), acn.shape)

    return np.array_equal(acn, expected)


def _convert_to_standard_definition(
        data,
        normalization,
        channel_convention,
        caxis_spherical_harmonics=(-1, )):
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
    caxis_spherical_harmonics : tuple
        Specifies which axis of data holds the spherical harmonic coefficients.
        Negative indexing, i.e., interpreted relative to the end of the array.

    Returns
    -------
    numpy.ndarray
        Spherical harmonic data following the standard definition (N3D, ACN).
    """

    # convert channel axes to axes in data
    axes = tuple(x - 1 for x in caxis_spherical_harmonics)

    data = renormalize(
        data, channel_convention, normalization,
        "N3D", axis=axes)

    data = change_channel_convention(
        data, channel_convention, "ACN", axis=axes)

    return data


def _convert_from_standard_definition(
        data,
        normalization,
        channel_convention,
        caxis_spherical_harmonics=(-1, )):
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
    caxis_spherical_harmonics : tuple
        Specifies which axis of data holds the spherical harmonic coefficients.
        Negative indexing, i.e., interpreted relative to the end of the array.

    Returns
    -------
    numpy.ndarray
        Spherical harmonic data according to the desired definition.
    """

    # convert channel axes to axes in data
    axes = tuple(x - 1 for x in caxis_spherical_harmonics)

    data = renormalize(
        data, "ACN", "N3D", normalization, axis=axes)

    data = change_channel_convention(
        data, "ACN", channel_convention, axis=axes)

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
    caxis_spherical_harmonics : int, tuple
        Specifies which channel axis (or axes) of data holds the spherical
        harmonic coefficients. A tuple of integers can be used to specify
        multiple spherical harmonics channel axes. Each value must be an
        integer smaller than or equal to -1. The default -1 refers to the last
        channel axis; a value of -2 would refer to the second last channel
        axis. See
        https://pyfar-gallery.readthedocs.io/en/latest/gallery/interactive/pyfar_audio_objects.html#Signal-cshape,-length,-and-caxis
        for more details.

    """

    def __init__(self, basis_type, normalization, channel_convention,
                 condon_shortley, caxis_spherical_harmonics, cshape):

        # must be set first, because n_max is required for checking the
        # channel convention and normalization
        self._is_sh_channel, self._acn = _init_spherical_harmonic_channels(
            cshape, caxis_spherical_harmonics)

        _SphericalHarmonicBase.__init__(
            self,
            basis_type,
            normalization,
            channel_convention,
            condon_shortley)

    @property
    def n_max(self):
        """Get the spherical harmonic order."""
        return int(np.sqrt(np.max(self._acn) + 1) - 1)

    @property
    def caxis_spherical_harmonics(self):
        """Get the spherical harmonic channel axes.

        The spherical harmonic channel axes are determined from the ambisonic
        channel numbers (ACN) of the channels, which are tracked when the
        object is reshaped, flattened, or transposed. A channel axis holds
        spherical harmonic coefficients if all its channels are spherical
        harmonic channels and the ACNs along the axis are
        ``0, 1, 2, ..., (N+1)^2-1``.

        Raises
        ------
        ValueError
            If the spherical harmonic coefficients are not contained in
            separate channel axes, e.g., after flattening the object. This can
            be resolved by reshaping the object.
        """
        is_sh_channel, acn = self._get_spherical_harmonic_channels()
        cdim = is_sh_channel.ndim

        if not np.all(is_sh_channel):
            raise ValueError(
                "The object contains channels that are not spherical "
                "harmonic channels.")

        caxis_spherical_harmonics = []
        for idx in range(acn.shape[-1]):
            # find channel axes holding valid ACNs that are not yet assigned
            candidates = [
                axis - cdim for axis in range(cdim)
                if _is_spherical_harmonic_axis(acn[..., idx], axis)
                and axis - cdim not in caxis_spherical_harmonics]

            if not candidates:
                raise ValueError(
                    "The spherical harmonic coefficients are not contained "
                    "in a channel axis, e.g., because the object was "
                    "flattened or reshaped. This can be resolved by "
                    "reshaping the object to a cshape that holds the "
                    "spherical harmonic coefficients in separate channel "
                    f"axes. The current cshape is {is_sh_channel.shape}.")

            # Multiple candidates can only occur for channel axes of size one
            # (n_max=0). In this case the axes are interchangeable and the
            # last one is used.
            caxis_spherical_harmonics.append(candidates[-1])

        return tuple(caxis_spherical_harmonics)

    def _get_spherical_harmonic_channels(self):
        """Return the arrays tracking the spherical harmonic channels.

        The arrays are broadcasted to the current cshape if required, e.g.,
        if the cshape of the data was changed by
        :py:func:`pyfar.utils.broadcast_cshape`.
        """
        is_sh_channel, acn = self._is_sh_channel, self._acn

        if hasattr(self, '_data') and is_sh_channel.shape != self.cshape:
            try:
                is_sh_channel = np.broadcast_to(is_sh_channel, self.cshape)
                acn = np.broadcast_to(acn, self.cshape + (acn.shape[-1], ))
            except ValueError as e:
                raise ValueError(
                    "The spherical harmonic channels of cshape "
                    f"{is_sh_channel.shape} do not match the cshape "
                    f"{self.cshape} of the data.") from e

        return is_sh_channel, acn

    def _update_spherical_harmonic_channels(self, caxis_spherical_harmonics):
        """Re-initialize the spherical harmonic channels if the cshape of the
        data changed, e.g., after setting the time or frequency data.
        """
        if self._is_sh_channel.shape != self.cshape:
            self._is_sh_channel, self._acn = \
                _init_spherical_harmonic_channels(
                    self.cshape, caxis_spherical_harmonics)

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

    def reshape(self, newshape):
        """
        Return reshaped copy of the SphericalHarmonicAudio object.

        The spherical harmonic channels are reshaped together with the data.
        If the spherical harmonic coefficients are not contained in separate
        channel axes after reshaping, e.g., ``cshape=(2, 4)`` is reshaped to
        ``cshape=(4, 2)``, the object can still be used as a container but
        :py:attr:`caxis_spherical_harmonics` and all operations requiring it
        raise an error until the object is reshaped to a valid cshape.

        Parameters
        ----------
        newshape : int, tuple
            new `cshape` of the audio object. One entry of newshape
            dimension can be ``-1``. In this case, the value is inferred
            from the remaining dimensions.

        Returns
        -------
        reshaped : SphericalHarmonicSignal, SphericalHarmonicFrequencyData, SphericalHarmonicTimeData
            reshaped copy of the audio object.

        Notes
        -----
        The number of samples and frequency bins always remains the same.

        """  # noqa: E501
        is_sh_channel, acn = self._get_spherical_harmonic_channels()
        reshaped = super().reshape(newshape)

        reshaped._is_sh_channel = is_sh_channel.reshape(reshaped.cshape)
        reshaped._acn = acn.reshape(reshaped.cshape + (acn.shape[-1], ))

        return reshaped

    def transpose(self, *axes):
        """Transpose time/frequency data and return copy of the
           SphericalHarmonicAudio object.

        The spherical harmonic channels are transposed together with the data.

        Parameters
        ----------
        axes : empty, ``None``, iterable of ints, or n ints
            Define how the
            :doc:`caxes<gallery:gallery/interactive/pyfar_audio_objects>`
            are ordered in the transposed audio object.
            Note that the last dimension of the data in the audio object
            always contains the time samples or frequency bins and can not
            be transposed.

            empty (default) or ``None``
                reverses the order of ``self.caxes``.
            iterable of ints
                `i` in the `j`-th place of the interable means
                that the `i`-th caxis becomes transposed object's `j`-th caxis.
            n ints
                same as 'iterable of ints'.
        """
        is_sh_channel, acn = self._get_spherical_harmonic_channels()
        transposed = super().transpose(*axes)

        # get transposed axes in the same way as pyfar
        cdim = len(self.cshape)
        if hasattr(axes, '__iter__'):
            axes = axes[0] if len(axes) == 1 else axes
        if axes is None or len(axes) == 0:
            axes = tuple(range(cdim))[::-1]
        else:
            axes = tuple(a % cdim if a < 0 else a for a in axes)

        transposed._is_sh_channel = is_sh_channel.transpose(axes)
        transposed._acn = acn.transpose(*axes, cdim)

        return transposed


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
        samples and the second to last the spherical harmonic coefficients. If
        the raw data have more then 3 dimensions `caxis_spherical_harmonics`
        defines the axis holding the spherical harmonic coefficients. The
        default is -1 (last channel axis). Accordingly, the default
        data shape follows the 'C' memory layout, e.g. data of
        ``shape = (1, 4, 1024)`` has 1 channel with 4 spherical harmonic
        coefficients with 1024 samples each, and
        a `caxis_spherical_harmonics` of -1. The data can be ``int``, ``float``
        or ``complex``. Data of type ``int`` is converted to ``float``.
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
    caxis_spherical_harmonics : int, tuple
        Specifies which channel axis (or axes) of data holds the spherical
        harmonic coefficients. A tuple of integers can be used to specify
        multiple spherical harmonics channel axes. Each value must be an
        integer smaller than or equal to -1. The default -1 refers to the last
        channel axis; a value of -2 would refer to the second last channel
        axis. See
        https://pyfar-gallery.readthedocs.io/en/latest/gallery/interactive/pyfar_audio_objects.html#Signal-cshape,-length,-and-caxis
        for more details.
    """

    def __init__(self, data, times, basis_type, normalization,
                 channel_convention, condon_shortley, comment="",
                 is_complex=False, caxis_spherical_harmonics=-1):

        if not is_complex and basis_type == 'complex':
            raise ValueError(
                "Complex spherical harmonic basis requires "
                "complex time data. Set is_complex=True.")

        if isinstance(caxis_spherical_harmonics, int):
            caxis_spherical_harmonics = (caxis_spherical_harmonics, )

        data = _atleast_3d_first_dimension(data)
        _assert_valid_caxis_spherical_harmonics(
            data.ndim, caxis_spherical_harmonics)

        _assert_valid_number_of_sh_channels(
            data.shape, caxis_spherical_harmonics)

        _SphericalHarmonicAudio.__init__(
            self, basis_type, normalization, channel_convention,
            condon_shortley,
            caxis_spherical_harmonics=caxis_spherical_harmonics,
            cshape=data.shape[:-1])

        TimeData.__init__(self, data=data, times=times, comment=comment,
                          is_complex=is_complex)

    @classmethod
    def from_definition(
            cls, sh_definition, data, times, comment="", is_complex=False,
            caxis_spherical_harmonics=-1):
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
            samples and the second to last the spherical harmonic coefficients.
            If the raw data have more then 3 dimensions
            `caxis_spherical_harmonics` defines the axis holding the spherical
            harmonic coefficients. The default is -1 (last channel axis).
            Accordingly, the default data shape follows the 'C' memory
            layout, e.g. data of ``shape = (1, 4, 1024)`` has 1 channel with 4
            spherical harmonic coefficients with 1024 samples each, and a
            `caxis_spherical_harmonics` of -1. The data can be ``int``,
            ``float`` or ``complex``. Data of type ``int`` is converted to
            ``float``.
        times : array, double
            Times in seconds at which the data is sampled. The number of times
            must match the size of the last dimension of `data`, i.e.,
            ``data.shape[-1]``.
        comment : str
            A comment related to `data`. The default is ``None``.
        is_complex : bool, optional
            A flag which indicates if the time data are real or complex-valued.
            The default is ``False``.
        caxis_spherical_harmonics : int, tuple
            Specifies which channel axis (or axes) of data holds the spherical
            harmonic coefficients. A tuple of integers can be used to specify
            multiple spherical harmonics channel axes. Each value must be an
            integer smaller than or equal to -1. The default -1 refers to the
            last channel axis; a value of -2 would refer to the second last
            channel axis. See
            https://pyfar-gallery.readthedocs.io/en/latest/gallery/interactive/pyfar_audio_objects.html#Signal-cshape,-length,-and-caxis
            for more details.
        """
        return cls(data, times,
                   basis_type=sh_definition.basis_type,
                   normalization=sh_definition.normalization,
                   channel_convention=sh_definition.channel_convention,
                   condon_shortley=sh_definition.condon_shortley,
                   comment=comment, is_complex=is_complex,
                   caxis_spherical_harmonics=caxis_spherical_harmonics)

    @property
    def time(self):
        """Return or set the time data."""

        return _convert_from_standard_definition(
            TimeData.time.fget(self), self.normalization,
            self.channel_convention, self.caxis_spherical_harmonics)

    @time.setter
    def time(self, value):
        """Return or set the time data."""
        value = _atleast_3d_first_dimension(value)
        caxis_spherical_harmonics = self.caxis_spherical_harmonics

        _assert_valid_number_of_sh_channels(
            value.shape, caxis_spherical_harmonics)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention,
            caxis_spherical_harmonics)
        TimeData.time.fset(self, value)
        self._update_spherical_harmonic_channels(
            caxis_spherical_harmonics)


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
        Raw data in the frequency domain. The data should have at least 2
        dimensions, with the last dimension representing the frequency domain
        bins and the second to last the spherical harmonic coefficients. If
        the raw data have more then 3 dimensions `caxis_spherical_harmonics`
        defines the axis holding the spherical harmonic coefficients. The
        default is -1 (last channel axis). Accordingly, the default
        data shape follows the 'C' memory layout, e.g. data of
        ``shape = (1, 4, 1024)`` has 1 channel with 4 spherical harmonic
        coefficients with 1024 frequency bins each, and a
        `caxis_spherical_harmonics` of -1. The data can be ``int``, ``float``
        or ``complex``. Data of type ``int`` is converted to ``float``.
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
    caxis_spherical_harmonics : int, tuple
        Specifies which channel axis (or axes) of data holds the spherical
        harmonic coefficients. A tuple of integers can be used to specify
        multiple spherical harmonics channel axes. Each value must be an
        integer smaller than or equal to -1. The default -1 refers to the last
        channel axis; a value of -2 would refer to the second last channel
        axis. See
        https://pyfar-gallery.readthedocs.io/en/latest/gallery/interactive/pyfar_audio_objects.html#Signal-cshape,-length,-and-caxis
        for more details.
    """

    def __init__(self, data, frequencies, basis_type, normalization,
                 channel_convention, condon_shortley, comment="",
                 caxis_spherical_harmonics=-1):

        if isinstance(caxis_spherical_harmonics, int):
            caxis_spherical_harmonics = (caxis_spherical_harmonics, )

        data = _atleast_3d_first_dimension(data)

        _assert_valid_caxis_spherical_harmonics(
            data.ndim, caxis_spherical_harmonics)

        _assert_valid_number_of_sh_channels(
            data.shape, caxis_spherical_harmonics)

        _SphericalHarmonicAudio.__init__(
            self, basis_type, normalization, channel_convention,
            condon_shortley,
            caxis_spherical_harmonics=caxis_spherical_harmonics,
            cshape=data.shape[:-1])

        FrequencyData.__init__(self, data=data, frequencies=frequencies,
                               comment=comment)

    @classmethod
    def from_definition(
            cls, sh_definition, data, frequencies, comment="",
            caxis_spherical_harmonics=-1):
        r"""
        Create a SphericalHarmonicFrequencyData class object from
        SphericalHarmonicDefinition object, data, and frequencies
        rate.

        Parameters
        ----------
        sh_definition : SphericalHarmonicDefinition
            The spherical harmonic definition.
        data : ndarray, double
            Raw data in the frequency domain. The data should have at least 2
            dimensions, with the last dimension representing the frequency
            domain bins and the second to last the spherical harmonic
            coefficients. If the raw data have more then 3 dimensions
            `caxis_spherical_harmonics` defines the axis holding the spherical
            harmonic coefficients. The default is -1 (last channel axis).
            Accordingly, the default data shape follows the 'C' memory
            layout, e.g. data of ``shape = (1, 4, 1024)`` has 1 channel with 4
            spherical harmonic coefficients with 1024 frequency bins each, and
            a `caxis_spherical_harmonics` of -1. The data can be ``int``,
            ``float`` or ``complex``. Data of type ``int`` is converted to
            ``float``.
        frequencies : array, double
            Frequencies of the data in Hz. The number of frequencies must match
            the size of the last dimension of `data`, i.e., ``data.shape[-1]``.
        comment : str
            A comment related to `data`. The default is ``None``.
        caxis_spherical_harmonics : int, tuple
            Specifies which channel axis (or axes) of data holds the spherical
            harmonic coefficients. A tuple of integers can be used to specify
            multiple spherical harmonics channel axes. Each value must be an
            integer smaller than or equal to -1. The default -1 refers to the
            last channel axis; a value of -2 would refer to the second last
            channel axis. See
            https://pyfar-gallery.readthedocs.io/en/latest/gallery/interactive/pyfar_audio_objects.html#Signal-cshape,-length,-and-caxis
            for more details.
        """
        return cls(data, frequencies,
                   basis_type=sh_definition.basis_type,
                   normalization=sh_definition.normalization,
                   channel_convention=sh_definition.channel_convention,
                   condon_shortley=sh_definition.condon_shortley,
                   comment=comment,
                   caxis_spherical_harmonics=caxis_spherical_harmonics)

    @property
    def freq(self):
        """Return or set the data in the frequency domain."""
        return _convert_from_standard_definition(
                    FrequencyData.freq.fget(self),
                    self.normalization,
                    self.channel_convention,
                    self.caxis_spherical_harmonics)

    @freq.setter
    def freq(self, value):
        """Return or set the data in the frequency domain."""
        value = _atleast_3d_first_dimension(value)
        caxis_spherical_harmonics = self.caxis_spherical_harmonics

        _assert_valid_number_of_sh_channels(
            value.shape, caxis_spherical_harmonics)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention,
            caxis_spherical_harmonics)

        FrequencyData.freq.fset(self, value)
        self._update_spherical_harmonic_channels(
            caxis_spherical_harmonics)


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
        samples/frequency domain bins and the second to last the spherical
        harmonic coefficients. If the raw data have more then 3 dimensions
        `caxis_spherical_harmonics` defines the axis holding the spherical
        harmonic coefficients. The default is -1 (last channel axis).
        Accordingly, the default data shape follows the 'C' memory layout,
        e.g. data of ``shape = (1, 4, 1024)`` has 1 channel with 4 spherical
        harmonic coefficients with 1024 samples or frequency bins each, and a
        `caxis_spherical_harmonics` of -1. The data can be ``int``, ``float``
        or ``complex``. Data of type ``int`` is converted to ``float``.
        Frequency is converted to ``complex`` and must be provided as single
        sided spectra, i.e., for all frequencies between 0 Hz and half the
        sampling rate.
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
    caxis_spherical_harmonics : int, tuple
        Specifies which channel axis (or axes) of data holds the spherical
        harmonic coefficients. A tuple of integers can be used to specify
        multiple spherical harmonics channel axes. Each value must be an
        integer smaller than or equal to -1. The default -1 refers to the last
        channel axis; a value of -2 would refer to the second last channel
        axis. See
        https://pyfar-gallery.readthedocs.io/en/latest/gallery/interactive/pyfar_audio_objects.html#Signal-cshape,-length,-and-caxis
        for more details.

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
                 caxis_spherical_harmonics=-1):

        if isinstance(caxis_spherical_harmonics, int):
            caxis_spherical_harmonics = (caxis_spherical_harmonics, )

        _assert_valid_caxis_spherical_harmonics(
            data.ndim, caxis_spherical_harmonics)

        data = _atleast_3d_first_dimension(data)

        _assert_valid_number_of_sh_channels(
            data.shape, caxis_spherical_harmonics)

        _SphericalHarmonicAudio.__init__(
            self, basis_type, normalization, channel_convention,
            condon_shortley,
            caxis_spherical_harmonics=caxis_spherical_harmonics,
            cshape=data.shape[:-1])

        Signal.__init__(self, data=data, sampling_rate=sampling_rate,
                        n_samples=n_samples, domain=domain, fft_norm=fft_norm,
                        comment=comment, is_complex=is_complex)

    @classmethod
    def from_definition(
            cls, sh_definition, data, sampling_rate, domain='time',
            fft_norm='none', comment="", is_complex=False,
            caxis_spherical_harmonics=-1):
        r"""
        Create a SphericalHarmonicSignal class object from
        SphericalHarmonicDefinition object, data, and sampling
        rate.

        Parameters
        ----------
        sh_definition : SphericalHarmonicDefinition
            The spherical harmonic definition.
        data : ndarray, double
            Raw data of the spherical harmonics signal in the time or
            frequency domain. The data should have at least 2 dimensions, with
            the last dimension representing the time domain
            samples/frequency domain bins and the second to last the spherical
            harmonic coefficients. If the raw data have more then 3 dimensions
            `caxis_spherical_harmonics` defines the axis holding the spherical
            harmonic coefficients. The default is -1 (last channel axis).
            Accordingly, the default data shape follows the 'C' memory
            layout, e.g. data of ``shape = (1, 4, 1024)`` has 1 channel with 4
            spherical harmonic coefficients with 1024 samples or frequency
            bins each, and a `caxis_spherical_harmonics` of -1. The data can be
            ``int``, ``float`` or ``complex``. Data of type ``int`` is
            converted to ``float``. Frequency is converted to ``complex`` and
            must be provided as single sided spectra, i.e., for all
            frequencies between 0 Hz and half the sampling rate.
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
        caxis_spherical_harmonics : int, tuple
            Specifies which channel axis (or axes) of data holds the spherical
            harmonic coefficients. A tuple of integers can be used to specify
            multiple spherical harmonics channel axes. Each value must be an
            integer smaller than or equal to -1. The default -1 refers to the
            last channel axis; a value of -2 would refer to the second last
            channel axis. See
            https://pyfar-gallery.readthedocs.io/en/latest/gallery/interactive/pyfar_audio_objects.html#Signal-cshape,-length,-and-caxis
            for more details.
        """
        return cls(data, sampling_rate,
                   basis_type=sh_definition.basis_type,
                   normalization=sh_definition.normalization,
                   channel_convention=sh_definition.channel_convention,
                   condon_shortley=sh_definition.condon_shortley,
                   domain=domain, fft_norm=fft_norm,
                   comment=comment, is_complex=is_complex,
                   caxis_spherical_harmonics=caxis_spherical_harmonics)

    @property
    def freq(self):
        """Return or set the data in the frequency domain."""

        return _convert_from_standard_definition(
                    Signal.freq.fget(self),
                    self.normalization,
                    self.channel_convention,
                    self.caxis_spherical_harmonics)

    @freq.setter
    def freq(self, value):
        """Return or set the data in the frequency domain."""
        value = _atleast_3d_first_dimension(value)
        caxis_spherical_harmonics = self.caxis_spherical_harmonics

        _assert_valid_number_of_sh_channels(
            value.shape, caxis_spherical_harmonics)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention,
            caxis_spherical_harmonics)

        Signal.freq.fset(self, value)
        self._update_spherical_harmonic_channels(
            caxis_spherical_harmonics)

    @property
    def freq_raw(self):
        """Return or set the frequency domain data without normalization."""

        return _convert_from_standard_definition(
                    Signal.freq_raw.fget(self),
                    self.normalization,
                    self.channel_convention,
                    self.caxis_spherical_harmonics)

    @freq_raw.setter
    def freq_raw(self, value):
        """Return or set the frequency domain data without normalization."""
        value = _atleast_3d_first_dimension(value)
        caxis_spherical_harmonics = self.caxis_spherical_harmonics

        _assert_valid_number_of_sh_channels(
            value.shape, caxis_spherical_harmonics)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention,
            caxis_spherical_harmonics)

        Signal.freq_raw.fset(self, value)
        self._update_spherical_harmonic_channels(
            caxis_spherical_harmonics)

    @property
    def time(self):
        """Return or set the time data."""

        return _convert_from_standard_definition(
                    Signal.time.fget(self),
                    self.normalization,
                    self.channel_convention,
                    self.caxis_spherical_harmonics)

    @time.setter
    def time(self, value):
        """Return or set the time data."""
        value = _atleast_3d_first_dimension(value)
        caxis_spherical_harmonics = self.caxis_spherical_harmonics

        _assert_valid_number_of_sh_channels(
            value.shape, caxis_spherical_harmonics)

        value = _convert_to_standard_definition(
            value, self.normalization, self.channel_convention,
            caxis_spherical_harmonics)

        Signal.time.fset(self, value)
        self._update_spherical_harmonic_channels(
            caxis_spherical_harmonics)
