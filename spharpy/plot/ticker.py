from matplotlib.ticker import Locator, Formatter
import spharpy
import numpy as np


class SphericalHarmonicLocator(Locator):

    def __init__(self, format='n', offset=0, step=1):
        self._offset = offset
        self._format = format
        self._step = step
        super().__init__()

    def __call__(self):
        dmin, dmax = self.axis.get_data_interval()

        return self.tick_values(dmin, dmax)

    def tick_values(self, vmin, vmax):
        n_min, m_min = spharpy.spherical.acn2nm(vmin)
        n_max, m_max = spharpy.spherical.acn2nm(vmax)

        orders = np.arange(n_min, n_max+1, step=self._step)

        if self._format == 'n':
            return spharpy.spherical.nm2acn(orders, orders) + self._offset
        elif self._format == 'n0':
            return spharpy.spherical.nm2acn(
                orders, np.zeros_like(orders)) + self._offset
        elif self._format == 'acn':
            return np.arange(vmin, vmax) + self._offset


class SphericalHarmonicFormatter(Formatter):
    """Formatter for ticks representing axes with spherical harmonics.
    Parameters
    ----------
    format : str
        The format used, can be either 'n', 'acn', or 'nm'
    """
    def __init__(self, format='n', offset=0, **kwargs):

        super().__init__(**kwargs)
        self.format = format
        self._offset = offset

    def __call__(self, x, pos=None):
        if self.format == 'n':
            string = spharpy.spherical.acn2nm(x)[0] + self._offset
        elif self.format == 'acn':
            string = x
        elif self.format == 'nm':
            n, m = spharpy.spherical.acn2nm(x)
            string = f'({n},{m})'
        else:
            raise ValueError(f'Unknown format {self.format}')

        return string
