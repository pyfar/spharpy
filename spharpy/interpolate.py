from scipy import interpolate as spinterpolate
import numpy as np


class SmoothSphereBivariateSpline(spinterpolate.SmoothSphereBivariateSpline):
    def __init__(self, sampling, data, w=None, s=1e-4, eps=1e-16):
        theta = sampling.elevation
        phi = sampling.azimuth
        if np.any(np.iscomplex(data)):
            raise ValueError("Complex data is not supported.")
        super().__init__(theta, phi, data, w, s, eps)

    def __call__(self, interp_grid, dtheta=0, dphi=0):
        theta = interp_grid.elevation
        phi = interp_grid.azimuth
        return super().__call__(
            theta, phi, dtheta=dtheta, dphi=dphi, grid=False)
