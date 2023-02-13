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
    coords = spharpy.samplings.icosahedron()
    data = np.cos(coords.azimuth)*np.sin(coords.elevation)
    spharpy.plot.balloon(coords, data, show=False)


def test_balloon_plot_phase():
    coords = spharpy.samplings.hyperinterpolation(10)
    data = np.cos(coords.azimuth)*np.sin(coords.elevation)
    spharpy.plot.balloon(coords, data, phase=True, show=False)


def test_contour_plot():
    coords = spharpy.samplings.hyperinterpolation(10)
    data = np.cos(coords.azimuth)*np.sin(coords.elevation)
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

    plt.close('all')


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

    plt.close('all')


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

    plt.close('all')


def test_contour():
    """Test if the plot executes without raising an exception
    """
    coords = spharpy.samplings.hyperinterpolation(10)
    data = np.cos(coords.azimuth)*np.sin(coords.elevation)
    plot.contour_map(coords, data)

    # test of auto detection of axes works
    ax = plt.axes()
    plot.contour(coords, data)

    # explicitly pass axes
    plot.contour(coords, data, ax=ax)

    plt.close('all')


def test_plot_voronoi_sphere():
    """Test if the plot executes without raising an exception
    """
    coords = spharpy.samplings.hyperinterpolation(4)
    plot.voronoi_cells_sphere(coords)

    # test of auto detection of axes works
    ax = plt.axes(projection='3d')
    plot.voronoi_cells_sphere(coords)

    # explicitly pass axes
    plot.voronoi_cells_sphere(coords, ax=ax)

    # pass axes with wrong projection
    ax = plt.axes()
    with pytest.raises(ValueError, match='3d'):
        plot.voronoi_cells_sphere(coords, ax=ax)

    # current axis with wrong projection
    with pytest.raises(ValueError, match='3d'):
        plot.voronoi_cells_sphere(coords)

    plt.close('all')


def test_pcolor_sphere():
    coords = spharpy.samplings.hyperinterpolation(10)
    data = np.cos(coords.azimuth)*np.sin(coords.elevation)
    plot.pcolor_sphere(coords, data, colorbar=True)

    # test of auto detection of axes works
    ax = plt.axes(projection='3d')
    plot.pcolor_sphere(coords, data)

    # explicitly pass axes
    plot.pcolor_sphere(coords, data, ax=ax)

    # pass axes with wrong projection
    ax = plt.axes()
    with pytest.raises(ValueError, match="'3d'"):
        plot.pcolor_sphere(coords, data, ax=ax)

    plt.close('all')


def test_balloon_wireframe():
    coords = spharpy.samplings.hyperinterpolation(10)
    data = np.cos(coords.azimuth)*np.sin(coords.elevation)
    plot.balloon_wireframe(coords, data)

    # test of auto detection of axes works
    ax = plt.axes(projection='3d')
    plot.balloon_wireframe(coords, data)

    # explicitly pass axes
    plot.balloon_wireframe(coords, data, ax=ax)

    # pass axes with wrong projection
    ax = plt.axes()
    with pytest.raises(ValueError, match="'3d'"):
        plot.balloon_wireframe(coords, data, ax=ax)

    plt.close('all')


def test_balloon():
    coords = spharpy.samplings.hyperinterpolation(10)
    data = np.cos(coords.azimuth)*np.sin(coords.elevation)
    plot.balloon(coords, data)

    # test of auto detection of axes works
    ax = plt.axes(projection='3d')
    plot.balloon(coords, data)

    # explicitly pass axes
    plot.balloon(coords, data, ax=ax)

    # pass axes with wrong projection
    ax = plt.axes()
    with pytest.raises(ValueError, match="'3d'"):
        plot.balloon(coords, data, ax=ax)

    plt.close('all')
