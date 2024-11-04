from pyfar import Signal
from spharpy.spherical import n3d_to_maxn, n3d_to_sn3d_norm
from spharpy.spherical import fuma_to_nm, acn_to_nm, nm_to_acn


class SphericalHarmonicSignal(Signal):
    """Create audio object with spherical harmonics coefficients in time or
    frequency domain.

    Objects of this class contain spherical harmonics coefficients which are
    directly convertible between time and frequency domain (equally spaced
    samples and frequency bins), the channel conventions `acn` and `fuma`, as well
    as the normalizations `n3d`, `sn3d`, or `maxn`.

    """
    def __init__(
            self,
            data,
            sampling_rate,
            n_sh,
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
            Raw data of the spherical harmonics signal in the time or
            frequency domain. The memory layout of data is 'C'. E.g.
            data of ``shape = (3, 4, 1024)`` has 3 channels with 4
            spherical harmonics coefficients with 1024 samples or frequency
            bins each. Time data is converted to ``float``. Frequency is
            converted to ``complex`` and must be provided as single
            sided spectra, i.e., for all frequencies between 0 Hz and
            half the sampling rate.
        sampling_rate : double
            Sampling rate in Hz
        n_sh : int
            Maximum spherical harmonic order. Has to match the number of
            coefficients, such that the number of coefficients
            >= (n_max + 1) ** 2.
        basis_type : str
            Type of spherical harmonic basis, either ``'complex'`` or
            ``'real'``. The default is ``'complex'``.
        normalization : str
            Normalization convention, either ``'n3d'``, ``'maxN'`` or
            ``'sn3d'``. The default is ``'n3d'``.
            (maxN is only supported up to 3rd order)
        channel_convention : str
            Channel ordering convention, either ``'acn'`` or ``'fuma'``.
            The default is ``'acn'``.
            (FuMa is only supported up to 3rd order)
        phase_convention : str
            Either 'condon-shortley', or 'ambix'.
            TODO: discuss ...
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
        .. [#] E.G. Williams, "Fourier Acoustics", (1999), Academic Press
        .. [#] B. Rafely, "Fundamentals of Spherical Array Processing", (2015),
               Springer-Verlag
        .. [#] F. Zotter, M. Frank, "Ambisonics A Practical 3D Audio Theory
               for Recording, Studio Production, Sound Reinforcement, and
               Virtual Reality", (2019), Springer-Verlag

        """

        self._init_n_sh(n_sh, data)
        self._init_basis_type(basis_type, is_complex)
        self._init_normalization(normalization)
        self._init_channel_convention(channel_convention)
        self._phase_convention = phase_convention

        if normalization in ['sn3d', 'n3d', 'maxN']:
            self._normalization = normalization
        else:
            raise ValueError("Invalid normalization, has to be 'sn3d', "
                             f"'n3d', or 'maxN, but is {normalization}")

        if channel_convention in ['acn', 'fuma']:
            self._channel_convention = channel_convention
        else:
            raise ValueError("Invalid channel convention, has to be 'acn' "
                             f"or 'fuma', but is {channel_convention}")

        Signal.__init__(self, data, sampling_rate=sampling_rate,
                        n_samples=n_samples, domain=domain, fft_norm=fft_norm,
                        comment=comment, is_complex=is_complex)

    @property
    def n_sh(self):
        return self.n_sh

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
        if value not in ["acn", "fuma"]:
            raise ValueError("Invalid channel convention, has to be 'acn' "
                             f"or 'fuma', but is {value}")

        if self.channel_convention is not value:
            self._change_channel_convention()

    @property
    def phase_convention(self):
        return self._phase_convention

    def _init_n_sh(self, value, data):
        """Set the spherical harmonic order."""
        if value < 0:
            raise ValueError("n_sh must be a positive integer")
        if value % 1 != 0:
            raise ValueError("n_sh must be an integer value")
        if not data.shape[-2] >= (value + 1) ** 2:
            raise ValueError('Data has to few sh coefficients '
                             'for n_sh = {n_sh}.')
        self._n_sh = int(value)

    def _init_basis_type(self, value, is_complex):
        """Set the basis type."""
        if value == 'complex' and not is_complex:
            raise ValueError('Data are real-valued while '
                             'spherical harmonics bases are complex-valued.')

        if value not in ["complex", "real"]:
            raise ValueError("Invalid basis type, only "
                             "'complex' and 'real' are supported")
        self._basis_type = value

    def _init_normalization(self, value):
        """Set the normalization convention."""
        if value not in ["n3d", "maxN", "sn3d"]:
            raise ValueError("Invalid normalization, has to be 'sn3d', "
                             f"'n3d', or 'maxN, but is {value}")
        self._normalization = value

    def _init_channel_convention(self, value):
        """Set the channel order convention."""
        if value not in ["acn", "fuma"]:
            raise ValueError("Invalid channel convention, has to be 'acn' "
                             f"or 'fuma', but is {value}")
        self._channel_convention = value

    def _renormalize(self, value):
        if value not in ["n3d", "maxN", "sn3d"]:
            raise ValueError("Invalid normalization, has to be 'sn3d', "
                             f"'n3d', or 'maxN, but is {value}")
        acn = range(0, (self.n_max + 1) ** 2)

        if self.channel_convention == "fuma":
            orders, degrees = fuma_to_nm(acn)
        else:
            orders, degrees = acn_to_nm(acn)

        if self._normalization == 'n3d':
            if value == "sn3d":
                self._data[:, :, ...] *= \
                    n3d_to_sn3d_norm(degrees, orders)
            elif value == "maxN":
                self._data[:, :, ...] *= \
                    n3d_to_maxn(acn)

        if self._normalization == 'sn3d':
            # convert to sn3d
            self._data[:, :, :] /= \
                    n3d_to_sn3d_norm(degrees, orders)
            if value == "maxN":
                self._data[:, acn, :] *= n3d_to_maxn(acn)

        if self._normalization == 'maxN':
            # convert to n3d
            self._data[:, acn, :] /= \
                    n3d_to_maxn(acn)
            if value == "sn3d":
                self._data[:, acn, :] *= \
                    n3d_to_sn3d_norm(acn)

    def _change_channel_convention(self):
        n_coeffs = (self.n_sh + 1) ** 2
        if self._channel_convention == 'acn':
            n, m = acn_to_nm(n_coeffs)
            #  idx = nm_to_fuma(n, m)
            raise NotImplementedError('not implemented')
        elif self._channel_convention == 'fuma':
            n, m = fuma_to_nm(n_coeffs)
            idx = nm_to_acn(n, m)

        self._data = self._data[:, idx, ...]
