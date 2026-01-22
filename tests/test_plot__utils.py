import pytest
import matplotlib.pyplot as plt
from spharpy.plot._utils import _prepare_plot


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

    if ax is None and projection is not None:
        assert ax.name == projection
