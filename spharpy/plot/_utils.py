"""Private utility functions for plot module."""

import numpy as np
import matplotlib.pyplot as plt


def _prepare_plot(ax=None, projection=None):
    """
    Returns a figure to plot on.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or list, tuple or ndarray of maplotlib.axes.Axes
        Axes to plot on. The default is None in which case the axes are
        obtained from the current figure. A new figure is created if it does
        not exist.
    projection : str, optional
        Type of projection for the axes. This is only applied if new axes are
        created. Default is ``None`` (2D projection). See
        `matplotlib.projections <https://matplotlib.org/stable/api/projections_api.html>`_
        for more information on projections.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Returns the active figure.
    ax : maptlotlib.axes.Axes
        Returns the current axes.
    """  # noqa: E501
    if ax is None:
        # get current figure or create a new one
        fig = plt.gcf()
        if fig.axes:
            ax = plt.gca()
        else:
            ax = plt.axes(projection=projection)

    else:
        # get figure from axis
        # ax can be array or tuple of two ax objects
        # first axis for the plot, second axis for colorbar placement
        if isinstance(ax, np.ndarray):
            fig = ax.flatten()[0].figure
        elif isinstance(ax, (list, tuple)):
            fig = ax[0].figure
        else:
            fig = ax.figure

    return fig, ax
