from pyfar import Signal
import numpy as np
from spharpy.spherical import SphericalHarmonics
import warnings


class AmbisonicsSignal(Signal):
    """Class for ambisonics signals.

    Objects of this class contain data which is directly convertible between
    time and frequency domain (equally spaced samples and frequency bins). The
    data is always real valued in the time domain and complex valued in the
    frequency domain.

    """
    def __init__(
            self,
            data,
            sampling_rate,
            spherical_harmonics,
            n_samples=None,
            domain='time',
            fft_norm='none',
            comment="",
            is_complex=False):
        """Create Ambisonicssignal with data, and sampling rate.

        Parameters
        ----------
        data : ndarray, double
            Raw data of the signal in the time or frequency domain. The memory
            layout of data is 'C'. E.g. data of ``shape = (3, 2, 1024)`` has
            3 x 2 channels with 1024 samples or frequency bins each. Time data
            is converted to ``float``. Frequency is converted to ``complex``
            and must be provided as single sided spectra, i.e., for all
            frequencies between 0 Hz and half the sampling rate.
        sampling_rate : double
            Sampling rate in Hz
        sh_kind : str
            Real or complex valued SH bases ``'real'``, ``'complex'``
        n_samples : int, optional
            Number of samples of the time signal. Required if domain is
            ``'freq'``. The default is ``None``, which assumes an even number
            of samples if the data is provided in the frequency domain.
        domain : ``'time'``, ``'freq'``, optional
            Domain of data. The default is ``'time'``
        fft_norm : str, optional
            The normalization of the Discrete Fourier Transform (DFT). Can be
            ``'none'``, ``'unitary'``, ``'amplitude'``, ``'rms'``, ``'power'``,
            or ``'psd'``. See :py:func:`~pyfar.dsp.fft.normalization` and [#]_
            for more information. The default is ``'none'``, which is typically
            used for energy signals, such as impulse responses.
        channel_order: str, optional
            The order in which the ambisonics channels are arranged.
            The default is ``'acn'``
        comment : str
            A comment related to `data`. The default is ``None``.

        To discuss: ambisonics channels always on first dimension?
        References
        ----------
        ..

        """

        if not type(spherical_harmonics) is SphericalHarmonics:
            raise ValueError("spherical_harmonics has be of type "
                             "`SphericalHarmonics`")

        self.spherical_harmonics = spherical_harmonics
        if self.spherical_harmonics.basis_type == 'complex' and not is_complex:
            raise ValueError('Data are real valued but spherical harmonics not')

        Signal.__init__(self, data, sampling_rate=sampling_rate,
                        n_samples=n_samples, domain=domain, fft_norm=fft_norm,
                        comment=comment, is_complex=is_complex)

    @property
    def spherical_harmonics(self):
        return self._spherical_harmonics

    @spherical_harmonics.setter
    def spherical_harmonics(self, value):
        self._spherical_harmonics = value

    @property
    def N(self):
        return self.spherical_harmonics.n_max

    @property
    def channel_convention(self):
        return self.spherical_harmonics.channel_convention


def sht(signal, coordinates, n_max, basis_type="real", domain=None, axis=0,
        channel_convention='acn', normalization='n3d',
        inv_type='pseudo_inverse'):
    """Compute the spherical harmonics transform at a certain order N

    Parameters
    ----------
    signal: Signal
        the signal for which the spherical harmonics transform is computed
    coordinates: :class:`spharpy.samplings.Coordinates`, :doc:`pf.Coordinates
        <pyfar:classes/pyfar.coordinates>`
        Coordinates on which the signal has been sampled
    n_max: integer
        Spherical harmonic order
    sh_kind: str
        Use real or complex valued SH bases ``'real'``, ``'complex'``
        default is ``'real'``
    axis: integer
        Axis along which the SH transform is computed

    Returns
    ----------
    AmbisonicsSignal

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

    if domain is None:
        domain = signal.domain

    #  coordinates = convert_coordinates(coordinates)

    if not signal.cshape[axis] == coordinates.csize:
        if coordinates.csize not in signal.cshape:
            raise ValueError("Signal shape does not match "
                             "number of coordinates.")
        else:
            axis = signal.cshape.index(coordinates.csize)
            warnings.warn(f"Compute SHT along axis={axis}.", UserWarning)

    spherical_harmonics = SphericalHarmonics(
        n_max=n_max, coords=coordinates, basis_type=basis_type,
        channel_convention=channel_convention,
        normalization=normalization, inverse_transform=inv_type)

    if domain == "time":
        data = signal.time
    elif domain == "freq":
        data = signal.freq
    else:
        raise ValueError("Domain should be ``'time'`` or ``'freq'`` but "
                         f"is {domain}.")

    Y_inv = spherical_harmonics.basis_inv  # [1] Eq. 3.34
    data_nm = np.tensordot(Y_inv, data, [1, axis])

    return AmbisonicsSignal(data=data_nm, domain=domain,
                            spherical_harmonics=spherical_harmonics,
                            sampling_rate=signal.sampling_rate,
                            fft_norm=signal.fft_norm,
                            comment=signal.comment)


def isht(ambisonics_signal, coordinates):
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
    _spherical_harmonics = ambisonics_signal.spherical_harmonics
    spherical_harmonics = SphericalHarmonics(
        _spherical_harmonics.n_max,
        coordinates,
        basis_type=_spherical_harmonics.basis_type,
        channel_convention=_spherical_harmonics.channel_convention,
        inverse_transform=_spherical_harmonics.inverse_transform,
        normalization=_spherical_harmonics.normalization)

    if ambisonics_signal.domain == "time":
        data_nm = ambisonics_signal.time
    else:
        data_nm = ambisonics_signal.freq

    # perform inverse transform
    data = np.tensordot(spherical_harmonics.basis, data_nm, [1, 0])

    return Signal(data, ambisonics_signal.sampling_rate,
                  fft_norm=ambisonics_signal.fft_norm,
                  comment=ambisonics_signal.comment)
