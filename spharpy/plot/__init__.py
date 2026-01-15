"""
Plotting functions for spherical data.

Troubleshooting - Overlapping Elements
--------------------------------------

A common issue with matplotlib 3D axes objects is that layout-options like
:func:`matplotlib.pyplot.tight_layout` or
:ref:`constrained layout <matplotlib:constrainedlayout_guide>`
don't always work correctly.

This can cause elements to overlap, such as colorbar and axes.
To prevent this, spharpy plot functions enable passing a list of axes for
the plot itself and the colorbar. The best way to handle space in the layout
is :class:`matplotlib.gridspec.GridSpec`.


Example
~~~~~~~~

.. plot::

    >>> import spharpy
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.gridspec import GridSpec
    >>>
    >>> coords = spharpy.samplings.equal_area(n_max=0, n_points=500)
    >>> data = np.sin(coords.colatitude) * np.cos(coords.azimuth)
    >>>
    >>> # Create figure and GridSpec
    >>> fig = plt.figure(figsize=(9, 7))
    >>> gs = GridSpec(nrows=1, ncols=2, width_ratios=[20, 1], wspace=0.3)
    >>>
    >>> # Create subplot axes for plot and colorbar
    >>> ax = fig.add_subplot(gs[0], projection='3d')
    >>> cax = fig.add_subplot(gs[1])
    >>>
    >>> # Plot
    >>> spharpy.plot.balloon(coords, data, ax = [ax, cax])
    >>> plt.show()
"""

from .spatial import (
    scatter,
    pcolor_map,
    pcolor_sphere,
    balloon,
    balloon_wireframe,
    voronoi_cells_sphere,
    contour,
    contour_map,
    MidpointNormalize,
)

from .cmap import phase_twilight


__all__ = [
    'scatter',
    'pcolor_map',
    'pcolor_sphere',
    'balloon',
    'balloon_wireframe',
    'voronoi_cells_sphere',
    'contour',
    'contour_map',
    'MidpointNormalize',
    'phase_twilight',
]
