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
    signal : Signal, TimeData, or FrequencyData
        the signal for which the spherical harmonics transform is computed
    spherical_harmonics : :class:`spharpy.SphericalHarmonics`
        Spherical harmonics object
    axis : integer or 'auto'
        Axis along which the SH transform is computed. If 'auto' the
        transformation is computed along the axis which matches the number
        of spherical samples of the spherical_harmonics basis

    Returns
    ----------
    sh_signal : SphericalHarmonicSignal, SphericalHarmonicsTimeData,
               or SphericalHarmonicsFrequencyData
                signal with spherical harmonics coefficients.
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
    if isinstance(signal, (Signal, TimeData)):
        data = signal.time
    elif isinstance(signal, FrequencyData):
        data = signal.freq
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

    # move spherical samples to -2
    data = np.moveaxis(data, axis, -2)

    # perform transform
    data_nm = np.tensordot(Y_inv, data, [1, axis])

    # move SH channels to -2
    data_nm = np.moveaxis(data_nm, 0, -2)

    # ensure result has 3 dimensions
    if len(data_nm.shape) < 3:
        data_nm = data_nm[np.newaxis, ...]

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
    sh_signal: SphericalHarmonicsSignal, SphericalHarmonicsTimeData, or
               SphericalHarmonicsFrequencyData
               The spherical harmonics signal for which the inverse spherical
               harmonics transform is computed
    coordinates: :class:`spharpy.samplings.Coordinates`, :doc:`pf.Coordinates
                 <pyfar:classes/pyfar.coordinates>`
                 Coordinates for which the inverse SH transform is computed

    Returns
    ----------
    Signal
    """
    if isinstance(sh_signal, (SphericalHarmonicSignal,
                              SphericalHarmonicTimeData)):
        data = sh_signal.time
    elif isinstance(sh_signal, SphericalHarmonicFrequencyData):
        data = sh_signal.freq
    else:
        raise ValueError("Input signal has to be SphericalHarmonicSignal, "
                         "SphericalHarmonicTimeData, or "
                         "SphericalHarmonicFrequencyData "
                         f"but is {type(sh_signal)}")

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

    # perform inverse transform
    data = np.tensordot(spherical_harmonics.basis, data, [1, -2])

    # and ensure that spherical samples are at second to last axis
    data = np.moveaxis(data, 0, -2)

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
