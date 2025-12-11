import numpy as np
from pyfar import Signal, TimeData, FrequencyData
from . import SphericalHarmonics
from . import SphericalHarmonicSignal
from . import SphericalHarmonicTimeData
from . import SphericalHarmonicFrequencyData


def sht(signal, spherical_harmonics, axis='auto'):
    """Compute the spherical harmonics transform

    Parameters
    ----------
    signal: Signal, TimeData, or FrequencyData
        the signal for which the spherical harmonics transform is computed
    spherical_harmonics: :class:`spharpy.SphericalHarmonics`
        Spherical harmonics object
    axis: integer
        Axis along which the SH transform is computed

    Returns
    ----------
    SphericalHarmonicSignal, SphericalHarmonicsTimeData,
    or SphericalHarmonicsFrequencyData

    References
    ----------

        [#] Rafaely, B. (2015). Fundamentals of Spherical Array Processing,
            (J. Benesty and W. Kellermann, Eds.) Springer Berlin Heidelberg,
            2nd ed., 196 pages. doi:10.1007/978-3-319-99561-8
        [#] Ramani Duraiswami, Dimitry N. Zotkin, and Nail A. Gumerov: "Inter-
            polation and range extrapolation of HRTFs." IEEE Int. Conf.
            Acoustics, Speech, and Signal Processing (ICASSP), Montreal,
            Canada, May 2004, p. 45-48, doi: 10.1109/ICASSP.2004.1326759.
    """
    if isinstance(signal, Signal):
        data = signal.time
        target_n = signal.n_samples
    elif isinstance(signal, TimeData):
        data = signal.time
        target_n = signal.n_samples
    elif isinstance(signal, FrequencyData):
        data = signal.freq
        target_n = signal.n_bins
    else:
        raise ValueError("Input signal must be a Signal, TimeData, or "
                         f"FrequencyData but is {type(signal)}")

    if not isinstance(spherical_harmonics, SphericalHarmonics):
        raise ValueError("spherical_harmonics must be SphericalHarmonics "
                         f"but is {type(spherical_harmonics)}")

    Y_inv = spherical_harmonics.basis_inv
    if axis == 'auto':
        axis = np.where(np.array(signal.cshape) == Y_inv.shape[1])[0]
        if len(axis) == 0:
            raise ValueError("No axes matches the number of spherical "
                             "harmonics basis functions")
        if len(axis) > 1:
            raise ValueError("Too many axis match the number of spherical "
                             "harmonics basis functions")
        axis = axis[0]

    if signal.cshape[axis] != Y_inv.shape[1]:
        raise ValueError("Spherical samples of provided axis does not match "
                         "the number of spherical harmonics basis functions.")

    # perform transform
    data_nm = np.tensordot(Y_inv, data, [1, axis])

    if len(data_nm.shape) < 3:
        data_nm = data_nm[np.newaxis, ...]

    # ensure that number of SH channels is at -2
    target_m = (spherical_harmonics.n_max+1)**2

    # find corresponding axes
    axis_m = next(i for i, dim in enumerate(data_nm.shape) if dim == target_m)
    axis_n = next(i for i, dim in enumerate(data_nm.shape)
                  if dim == target_n and i != axis_m)

    # create new shape
    new_axes = [
        i for i in range(len(data_nm.shape)) if i not in (axis_m, axis_n)
    ] + [axis_m, axis_n]

    data_nm = data_nm.transpose(*new_axes)

    if isinstance(signal, Signal):
        sh_signal = SphericalHarmonicSignal(
                    data=data_nm,
                    basis_type=spherical_harmonics.basis_type,
                    normalization=spherical_harmonics.normalization,
                    channel_convention=spherical_harmonics.channel_convention,
                    condon_shortley=spherical_harmonics.condon_shortley,
                    sampling_rate=signal.sampling_rate,
                    fft_norm=signal.fft_norm,
                    is_complex=signal.complex,
                    comment=signal.comment)
    elif isinstance(signal, TimeData):
        sh_signal = SphericalHarmonicTimeData(
                    data=data_nm,
                    times=signal.times,
                    basis_type=spherical_harmonics.basis_type,
                    normalization=spherical_harmonics.normalization,
                    channel_convention=spherical_harmonics.channel_convention,
                    condon_shortley=spherical_harmonics.condon_shortley,
                    comment=signal.comment,
                    is_complex=False)
    elif isinstance(signal, FrequencyData):
        sh_signal = SphericalHarmonicFrequencyData(
                    data=data_nm,
                    frequencies=signal.frequencies,
                    basis_type=spherical_harmonics.basis_type,
                    normalization=spherical_harmonics.normalization,
                    channel_convention=spherical_harmonics.channel_convention,
                    condon_shortley=spherical_harmonics.condon_shortley,
                    comment=signal.comment)

    return sh_signal


def isht(sh_signal, coordinates):
    """Compute the inverse spherical harmonics transform

    Parameters
    ----------
    sh_signal: Signal
        The spherical harmonics signal for which the inverse spherical
        harmonics transform is computed
    coordinates: :class:`spharpy.samplings.Coordinates`, :doc:`pf.Coordinates
                 <pyfar:classes/pyfar.coordinates>`
        Coordinates for which the inverse SH transform is computed

    Returns
    ----------
    Signal
    """

    # get spherical harmonics basis functions according to sh_signals
    # properties
    spherical_harmonics = SphericalHarmonics(
        sh_signal.n_max,
        coordinates=coordinates,
        basis_type=sh_signal.basis_type,
        channel_convention=sh_signal.channel_convention,
        normalization=sh_signal.normalization,
        inverse_method="pseudo_inverse",
        condon_shortley=sh_signal.condon_shortley)

    if isinstance(sh_signal, SphericalHarmonicSignal):
        data = sh_signal.time
    elif isinstance(sh_signal, SphericalHarmonicTimeData):
        data = sh_signal.time
    elif isinstance(sh_signal, SphericalHarmonicFrequencyData):
        data = sh_signal.freq
    else:
        raise ValueError("Input signal has to be SphericalHarmonicSignal, "
                         "SphericalHarmonicTimeData, or "
                         "SphericalHarmonicFrequencyData "
                         f"but is {type(sh_signal)}")

    # perform inverse transform
    data = np.tensordot(spherical_harmonics.basis, data, [1, -2])

    if isinstance(sh_signal, SphericalHarmonicSignal):
        signal = Signal(data,
                        sh_signal.sampling_rate,
                        fft_norm=sh_signal.fft_norm,
                        comment=sh_signal.comment,
                        is_complex=sh_signal.complex)
    elif isinstance(sh_signal, SphericalHarmonicTimeData):
        signal = TimeData(data=data,
                          times=sh_signal.times,
                          comment=sh_signal.comment,
                          is_complex=sh_signal.complex)
    else:
        signal = FrequencyData(data=data,
                               frequencies=sh_signal.frequencies,
                               comment=sh_signal.comment)
    return signal
