import pytest
import spharpy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from spharpy.plot._utils import _prepare_plot, _add_colorbar


@pytest.mark.parametrize(
    ("input_ax", "output_type", "projection"),
    [(None, plt.Axes, '3d'), (plt.gca(), plt.Axes, None),
     ([plt.gca(), plt.gca()], list, None)],
)
def test_prepare_plot(input_ax, output_type, projection):
    """
    Test output of :py:func:`~spharpy.plot._utils._prepare_plot`.
    """
    fig, ax = _prepare_plot(input_ax, projection)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, output_type)

    if isinstance(ax, list):
        assert all(isinstance(ax_, plt.Axes) for ax_ in input_ax)

    if ax is None:
        assert ax.name == projection


@pytest.mark.parametrize(
        ("colorbar", "ax", "return_type"),
        [(True, [plt.axes(), None], mpl.colorbar.Colorbar),
         (True, [plt.axes(), plt.axes()], mpl.colorbar.Colorbar),
         (False, [plt.axes(), None], None)],
)
def test_add_colorbar(colorbar, ax, return_type):
    """Test return type of :py:func:`~spharpy.plot._utils._add_colorbar`"""
    norm = mpl.colors.Normalize(vmin=0, vmax=10)
    cmap = plt.get_cmap('viridis')
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.gcf()
    cb = _add_colorbar(colorbar, fig, ax, mappable, None)

    if return_type is None:
        assert cb is None
    else:
        assert isinstance(cb, return_type)
