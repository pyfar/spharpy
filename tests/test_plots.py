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
    coords = spharpy.samplings.hyperinterpolation(10)

    ax = plt.axes(projection='3d')
    spharpy.plot.scatter(coords, ax=ax)

    ax = plt.axes()
    with pytest.raises(ValueError, match='3d'):
        spharpy.plot.scatter(coords, ax=ax)
