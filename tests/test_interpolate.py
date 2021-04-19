import spharpy
from spharpy.samplings import Coordinates
import spharpy.interpolate as interpolate
import numpy as np


def test_smooth_sphere_bivariate_spline_interpolation():
    n_max = 10
    sampling = spharpy.samplings.equalarea(
        n_max, n_points=500, condition_num=np.inf)
    Y = spharpy.spherical.spherical_harmonic_basis_real(n_max, sampling)
    y_vec = spharpy.spherical.spherical_harmonic_basis_real(
        n_max, Coordinates(1, 0, 0))
    data = Y @ y_vec.T

    data = np.sin(sampling.elevation)*np.sin(2*sampling.azimuth)

    interp_grid = spharpy.samplings.hyperinterpolation(35)

    data_grid = np.sin(interp_grid.elevation)*np.sin(2*interp_grid.azimuth)

    interpolator = interpolate.SmoothSphereBivariateSpline(
        sampling, data, s=1e-4)

    interp_data = interpolator(interp_grid)

    # check if error over entire sphere sufficiently small
    assert np.linalg.norm(np.abs(interp_data - data_grid)) / \
        np.linalg.norm(np.abs(data_grid)) < 1e-2
