import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from spharpy.plot._utils import _prepare_plot, _add_colorbar


@pytest.mark.parametrize(
    ("ax_case", "output_type", "projection"),
    [("none", plt.Axes, "3d"), ("single", plt.Axes, None),
     ("two", list, None)],
)
def test_prepare_plot(ax_case, output_type, projection):
    """
    Test output of :py:func:`~spharpy.plot._utils._prepare_plot`.
    """
    if ax_case == "none":
        input_ax = None
    elif ax_case == "single":
        _, input_ax = plt.subplots()
    else:
        _, axs = plt.subplots(1, 2)
        input_ax = [axs[0], axs[1]]

    fig, ax = _prepare_plot(input_ax, projection)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, output_type)

    if isinstance(ax, list):
        assert all(isinstance(ax_, plt.Axes) for ax_ in ax)

    if ax is None:
        assert ax.name == projection


@pytest.mark.parametrize(
    ("colorbar", "ax_case", "return_type"),
    [(True, "single", mpl.colorbar.Colorbar),
     (True, "two", mpl.colorbar.Colorbar), (False, "single", None)],
)
def test_add_colorbar(colorbar, ax_case, return_type):
    """Test return type of :py:func:`~spharpy.plot._utils._add_colorbar`."""
    if ax_case == "single":
        fig, ax0 = plt.subplots()
        ax = [ax0, None]
    else:
        fig, axs = plt.subplots(1, 2)
        ax = [axs[0], axs[1]]

    norm = mpl.colors.Normalize(vmin=0, vmax=10)
    cmap = plt.get_cmap('viridis')
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.gcf()
    cb = _add_colorbar(colorbar, fig, ax, mappable, None)

    if return_type is None:
        assert cb is None
    else:
        assert isinstance(cb, return_type)
