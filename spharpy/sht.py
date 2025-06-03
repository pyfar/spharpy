import numpy as np
from pyfar import Signal
import warnings
from . import SphericalHarmonicSignal


def sht(signal, coordinates, n_max, basis_type="real", axis=-2,
        channel_convention='acn', normalization='n3d', condon_shortley='auto',
        inverse_method='pseudo_inverse'):
    """Compute the spherical harmonics transform

    Parameters
    ----------
    signal: Signal
        the signal for which the spherical harmonics transform is computed
    coordinates: :class:`spharpy.samplings.Coordinates`, :doc:`pf.Coordinates
        <pyfar:classes/pyfar.coordinates>`
        Coordinates on which the signal has been sampled
    n_max: integer
        Spherical harmonic order
    basis_type: str
        Use real or complex valued SH bases ``'real'``, ``'complex'``
        default is ``'real'``
    axis: integer
        Axis along which the SH transform is computed
    channel_convention: 
    normalization:
    condon_shortley:
    inverse_method:

    Returns
    ----------
    SphericalHarmonicSignal

    References
    ----------

        [1] Rafaely, B. (2015). Fundamentals of Spherical Array Processing,
            (J. Benesty and W. Kellermann, Eds.) Springer Berlin Heidelberg,
            2nd ed., 196 pages. doi:10.1007/978-3-319-99561-8
        [2] Ramani Duraiswami, Dimitry N. Zotkin, and Nail A. Gumerov: "Inter-
            polation and range extrapolation of HRTFs." IEEE Int. Conf.
            Acoustics, Speech, and Signal Processing (ICASSP), Montreal,
            Canada, May 2004, p. 45-48, doi: 10.1109/ICASSP.2004.1326759.
    """

    if not signal.cshape[axis] == coordinates.csize:
        if coordinates.csize not in signal.cshape:
            raise ValueError("Signal shape does not match "
                             "number of coordinates.")
        else:
            axis = signal.cshape.index(coordinates.csize)
            warnings.warn(f"Compute SHT along axis={axis}.", UserWarning)

    spherical_harmonics = SphericalHarmonics(
        n_max=n_max, coordinates=coordinates, basis_type=basis_type,
        channel_convention=channel_convention,
        normalization=normalization, inverse_method=inverse_method,
        condon_shortley=condon_shortley)

    Y_inv = spherical_harmonics.basis_inv  # [1] Eq. 3.34
    data_nm = np.tensordot(Y_inv, data.time, [1, axis])

    # ensure that number of SH channels is at -2
    data_nm = data_nm.reshape((-1, (n_max+1)**2, signal.n_samples))

    return SphericalHarmonicSignal(data=data_nm,
                                   basis_type=basis_type,
                                   normalization=normalization,
                                   channel_convention=channel_convention,
                                   condon_shortley=spherical_harmonics.condon_shortley,
                                   sampling_rate=signal.sampling_rate,
                                   fft_norm=signal.fft_norm,
                                   comment=signal.comment)


def isht(sh_signal, coordinates):
    """Compute the inverse spherical harmonics transform at a certain order N

    Parameters
    ----------
    ambisonics_signal: Signal
        The ambisonics signal for which the inverse spherical harmonics
        transform is computed
    coordinates: :class:`spharpy.samplings.Coordinates`, :doc:`pf.Coordinates
                 <pyfar:classes/pyfar.coordinates>`
        Coordinates for which the inverse SH transform is computed
    """
    # get spherical harmonics basis functions of same type as the the
    # ambisonics signals but for the passed coordinates
    spherical_harmonics = SphericalHarmonics(
        sh_signal.n_max,
        coordinates=coordinates,
        basis_type=sh_signal.basis_type,
        channel_convention=sh_signal.channel_convention,
        normalization=sh_signal.normalization,
        condon_shortley=sh_signal.condon_shortley)

    # perform inverse transform
    data = np.tensordot(spherical_harmonics.basis, sh_signal.time, [1, -2])

    return Signal(data, ambisonics_signal.sampling_rate,
                  fft_norm=ambisonics_signal.fft_norm,
                  comment=ambisonics_signal.comment)
