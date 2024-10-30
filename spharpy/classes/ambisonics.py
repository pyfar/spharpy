from pyfar import Signal
from spharpy.spherical import n3d_to_maxn, n3d_to_sn3d_norm
from spharpy.spherical import fuma_to_nm, acn_to_nm, nm_to_acn


class SphericalHarmonicSignal(Signal):
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
            n_max,
            basis_type,
            normalization,
            channel_convention,
            phase_convention,
            n_samples=None,
            domain='time',
            fft_norm='none',
            comment="",
            is_complex=False):
        """Create SphericalHarmonicSignal with data, and sampling rate.

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
        n_max : int
            Maximum spherical harmonic order
        basis_type : str, optional
            Type of spherical harmonic basis, either ``'complex'`` or
            ``'real'``. The default is ``'complex'``.
        normalization : str, optional
            Normalization convention, either ``'n3d'``, ``'maxN'`` or
            ``'sn3d'``. The default is ``'n3d'``.
            (maxN is only supported up to 3rd order)
        channel_convention : str, optional
            Channel ordering convention, either ``'acn'`` or ``'fuma'``.
            The default is ``'acn'``.
            (FuMa is only supported up to 3rd order)
        phase_convention : bool, optional
            Whether to include the Condon-Shortley phase term.
            The default is True.
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
        comment : str
            A comment related to `data`. The default is ``None``.

        References
        ----------
        ..

        """
        # TODO: discuss if ambisonics channels always on first dimension?
        if not data.shape[1] >= (n_max + 1) ** 2:
            raise ValueError('Data has to few coefficients '
                             'for N = {n_max}.')

        self._n_max = n_max

        if basis_type == 'complex' and not is_complex:
            raise ValueError('Data are real-valued while '
                             'spherical harmonics bases are complex-valued.')

        self._basis_type = basis_type
        self._phase_convention = phase_convention

        self.normalization = normalization
        self.channel_convention = channel_convention

        Signal.__init__(self, data, sampling_rate=sampling_rate,
                        n_samples=n_samples, domain=domain, fft_norm=fft_norm,
                        comment=comment, is_complex=is_complex)

    @property
    def n_max(self):
        return self.n_max

    @property
    def basis_type(self):
        return self._basis_type

    @property
    def normalization(self):
        return self._normalization

    @normalization.setter
    def normalization(self, value):
        if self.normalization is not value:
            self._renormalize(self, value)

        self._normalization = value

    @property
    def channel_convention(self):
        return self._channel_convention

    @channel_convention.setter
    def channel_convention(self, value):
        if self.channel_convention is not value:
            self._change_channel_convention(value)

    @property
    def phase_convention(self):
        return self._phase_convention

    def _renormalize(self, normalization):
        acn = range(0, (self.n_max + 1) ** 2)

        if self.channel_convention == "fuma":
            orders, degrees = fuma_to_nm(acn)
        else:
            orders, degrees = acn_to_nm(acn)

        if self._normalization == 'n3d':
            if normalization == "sn3d":
                self._data[:, :, ...] *= \
                    n3d_to_sn3d_norm(degrees, orders)
            elif normalization == "maxN":
                self._data[:, :, ...] *= \
                    n3d_to_maxn(acn)

        if self._normalization == 'sn3d':
            # convert to sn3d
            self._data[:, :, :] /= \
                    n3d_to_sn3d_norm(degrees, orders)
            if normalization == "maxN":
                self._data[:, acn, :] *= n3d_to_maxn(acn)

        if self._normalization == 'maxN':
            # convert to n3d
            self._data[:, acn, :] /= \
                    n3d_to_maxn(acn)
            if normalization == "sn3d":
                self._data[:, acn, :] *= \
                    n3d_to_sn3d_norm(acn)

    def _change_channel_convention(self, value):
        n_coeffs = (self.n_max + 1) ** 2
        if self._channel_convention == 'acn':
            n, m = acn_to_nm(n_coeffs)
            #  idx = nm_to_fuma(n, m)
            raise NotImplementedError('not implemented')
        elif self._channel_convention == 'fuma':
            n, m = fuma_to_nm(n_coeffs)
            idx = nm_to_acn(n, m)

        self._data = self._data[:, idx, ...]
