import matplotlib as mpl
# Use Agg backend to prevent matplotlib from creating interactive plots. This
# has to be set before the importing matplotlib.pyplot. Use switch backend in
# case the wrong backend has already been set.
mpl.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import spharpy
import numpy as np
import pytest
from spharpy import plot


def test_balloon_plot_abs():
    n_max = 1
    coords = spharpy.samplings.icosahedron()
    doa = spharpy.samplings.Coordinates(1, 0, 0)
    y_vec = spharpy.spherical.spherical_harmonic_basis_real(n_max, doa)
    basis = spharpy.spherical.spherical_harmonic_basis_real(n_max, coords)
    data = np.squeeze(4*np.pi/(n_max+1)**2 * basis @ y_vec.T)
    spharpy.plot.balloon(coords, data, show=False)


def test_balloon_plot_phase():
    n_max = 1
    coords = spharpy.samplings.hyperinterpolation(10)
    doa = spharpy.samplings.Coordinates(1, 0, 0)
    y_vec = spharpy.spherical.spherical_harmonic_basis_real(n_max, doa)
    basis = spharpy.spherical.spherical_harmonic_basis_real(n_max, coords)
    data = np.squeeze(4*np.pi/(n_max+1)**2 * basis @ y_vec.T)
    spharpy.plot.balloon(coords, data, phase=True, show=False)


def test_contour_plot():
    n_max = 1
    coords = spharpy.samplings.hyperinterpolation(10)
    doa = spharpy.samplings.Coordinates(1, 0, 0)
    y_vec = spharpy.spherical.spherical_harmonic_basis_real(n_max, doa)
    basis = spharpy.spherical.spherical_harmonic_basis_real(n_max, coords)
    data = np.squeeze(4*np.pi/(n_max+1)**2 * basis @ y_vec.T)
    spharpy.plot.contour(coords, np.abs(data), show=False)


def test_scatter():
    """Test if the plot executes without raising an exception
    """
    coords = spharpy.samplings.hyperinterpolation(10)
    spharpy.plot.scatter(coords)

    # test of auto detection of axes works
    ax = plt.axes(projection='3d')
    spharpy.plot.scatter(coords)

    # explicitly pass axes
    spharpy.plot.scatter(coords, ax=ax)

    # pass axes with wrong projection
    ax = plt.axes()
    with pytest.raises(ValueError, match='3d'):
        spharpy.plot.scatter(coords, ax=ax)

    # current axis with wrong projection
    with pytest.raises(ValueError, match='3d'):
        spharpy.plot.scatter(coords)


def test_pcolor_map():
    """Test if the plot executes without raising an exception
    """
    coords = spharpy.samplings.hyperinterpolation(10)
    data = np.cos(coords.azimuth)*np.sin(coords.elevation)
    plot.pcolor_map(coords, data)

    # test of auto detection of axes works
    ax = plt.axes(projection='mollweide')
    plot.pcolor_map(coords, data)

    # explicitly pass axes
    plot.pcolor_map(coords, data, ax=ax)

    # pass axes with wrong projection
    ax = plt.axes()
    with pytest.raises(ValueError, match='Projection does not match'):
        plot.pcolor_map(coords, data, ax=ax)


def test_contour_map():
    """Test if the plot executes without raising an exception
    """
    coords = spharpy.samplings.hyperinterpolation(10)
    data = np.cos(coords.azimuth)*np.sin(coords.elevation)
    plot.contour_map(coords, data)

    # test of auto detection of axes works
    ax = plt.axes(projection='mollweide')
    plot.contour_map(coords, data)

    # explicitly pass axes
    plot.contour_map(coords, data, ax=ax)

    # pass axes with wrong projection
    ax = plt.axes()
    with pytest.raises(ValueError, match='Projection does not match'):
        plot.contour_map(coords, data, ax=ax)
