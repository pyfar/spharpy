from pyfar import Signal
import numpy as np


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
            sh_kind,
            n_samples=None,
            domain='time',
            fft_norm='none',
            channel_order='acn',
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
            The order in which the ambisonics channels are arraged.
            The default is ``'acn'``
        comment : str
            A comment related to `data`. The default is ``None``.

        To discuss: ambisonics channels always on first dimension?
        References
        ----------
        ..

        """

        # initialize signal specific parameters
        self._sampling_rate = sampling_rate
        if sh_kind in ["real", "complex"]:
            self._sh_kind = sh_kind
        else:
            raise ValueError("sh kind has to be ``real`` or ``complex``")
        self._channel_order = channel_order
        Signal.__init__(self, data, sampling_rate=sampling_rate,
                        n_samples=n_samples, domain=domain, fft_norm=fft_norm,
                        comment=comment, is_complex=is_complex)

    @property
    def N(self):
        return int(np.sqrt(self.cshape[0]-1))

    @property
    def sh_kind(self):
        return self._sh_kind

    @property
    def channel_order(self):
        return self._channel_order
