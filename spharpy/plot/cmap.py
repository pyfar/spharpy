from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np


def phase_twilight(lut=512):
    """Cyclic color map for displaying phase information. This is a modified
    version of the twilight color map from matplotlib.
    """
    lut = int(np.ceil(lut/4)*4)
    twilight = cm.get_cmap('twilight', lut=lut)

    twilight_r_colors = np.array(twilight.reversed().colors)

    roll_by = int(lut/4)
    phase_colors = np.roll(twilight_r_colors, -roll_by, axis=0)

    return ListedColormap(phase_colors)
