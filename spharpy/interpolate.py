from scipy import interpolate as spinterpolate
import numpy as np


class SmoothSphereBivariateSpline(spinterpolate.SmoothSphereBivariateSpline):
    def __init__(self, sampling, data, w=None, s=None, eps=1e-16):
        theta = sampling.elevation
        phi = sampling.azimuth
        if s is None:
            s = sampling.n_points
        if np.any(np.iscomplex(data)):
            raise ValueError("Complex data is not supported.")
        super().__init__(theta, phi, data, w, s, eps)

    def __call__(self, grid, **kwargs):
        theta = grid.elevation
        phi = grid.azimuth
        return super().__call__(theta, phi, **kwargs)
