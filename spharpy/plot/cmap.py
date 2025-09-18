"""cmap for displaying phase information."""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def phase_twilight(lut=512):
    """
    Cyclic color map for displaying phase information.

    This is a modified version of the twilight color map from matplotlib.

    Parameters
    ----------
    lut : int, optional
        Number of entries in the lookup table of colors for the colormap.
        Default is 512.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Colormap instance.
    """
    if not isinstance(lut, int) or lut <= 0:
        raise ValueError('lut must be a positive integer.')

    lut = int(np.ceil(lut/4)*4)
    twilight = plt.get_cmap('twilight', lut=lut)

    twilight_r_colors = np.array(twilight.reversed().colors)

    roll_by = int(lut/4)
    phase_colors = np.roll(twilight_r_colors, -roll_by, axis=0)

    return ListedColormap(phase_colors)
