"""Private utility functions for plot module."""

import numpy as np
import matplotlib.pyplot as plt


def _add_colorbar(colorbar, fig, ax, mappable, label):
    """
    Add colorbar to plot.

    Parameters
    ----------
    colorbar : bool
        Flag indicating if a colobar should be added to the plot
    fig : matplotlib.figure.Figure
        Figure to plot on.
    ax : list[matplotlib.axes.Axes]
        Either a list of two axes objects or a list with one axis and None
        object.
    mappable : matplotlib.cm.ScalarMappable
        The matplotlib.cm.ScalarMappable described by the colorbar.
    label : string
        colorbar label

    Returns
    -------
    cb : matplotlib.colorbar.Colorbar
        Returns matplotlib colorbar object.
    """
    if colorbar:
        if ax[1] is None:
            cb = fig.colorbar(mappable, ax=ax[0])
        else:
            cb = fig.colorbar(mappable, ax=ax[1])
        cb.set_label(label)
    else:
        cb = None

    return cb


def _prepare_plot(ax, projection=None):
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
        created. Default is ``None`` (2D projection).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Returns the active figure.
    ax : maptlotlib.axes.Axes
        Returns the current axes.
    """
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
