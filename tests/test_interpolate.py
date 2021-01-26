import spharpy
from spharpy.samplings import Coordinates
import spharpy.interpolate as interpolate
from scipy.interpolate import SmoothSphereBivariateSpline
import numpy as np

import matplotlib.pyplot as plt


def test_smooth_sphere_bivariate_spline_interpolation():
    n_max = 10
    sampling = spharpy.samplings.equalarea(20, n_points=500, condition_num=np.inf)
    Y = spharpy.spherical.spherical_harmonic_basis_real(n_max, sampling)
    y_vec = spharpy.spherical.spherical_harmonic_basis_real(
        n_max, Coordinates(1, 0, 0))
    data = Y @ y_vec.T

    data = np.sin(sampling.elevation)*np.sin(2*sampling.azimuth)

    interp_grid = spharpy.samplings.gaussian(15)

    weights = np.ones(sampling.n_points)
    interpolator = interpolate.SmoothSphereBivariateSpline(
        sampling, data, s=sampling.n_points)

    interpolator(interp_grid)
