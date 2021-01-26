from scipy import interpolate as spinterpolate
import numpy as np


class SmoothSphereBivariateSpline(spinterpolate.SmoothSphereBivariateSpline):
    """Smooth bivariate spline approximation in spherical coordinates.

    Note
    ----
    This is a wrapper for scipy's SmoothSphereBivariateSpline only using
    Coordinates objects as container for the azimuth and elevation angles.
    For detailed information see scipy's documentation.
    """
    def __init__(self, sampling, data, w=None, s=1e-4, eps=1e-16):
        """
        Parameters
        ----------
        sampling : Coordinates
            Coordinates object containing the positions for which the data
            is sampled
        data : array, float
            Array containing the data at the sampling positions. Has to be real
            valued.
        w : array, float
            Weighting coefficients
        s : float, 1e-4
            Smoothing factor > 0
        eps : float, 1e-16
            The eps valued to be considered for interpolator estimation.
            Depends on the used data type and numerical precision.
        """
        theta = sampling.elevation
        phi = sampling.azimuth
        if np.any(np.iscomplex(data)):
            raise ValueError("Complex data is not supported.")
        super().__init__(theta, phi, data, w, s, eps)

    def __call__(self, interp_grid, dtheta=0, dphi=0):
        """Evaluate the spline on a new sampling grid.

        Parameters
        ----------
        interp_grid : Coordinates
            Coordinates object containing a new set of points for which data
            is to be interpolated.
        dtheta : int, optional
            Order of theta derivative
        dphi : int, optional
            Order of phi derivative
        """
        theta = interp_grid.elevation
        phi = interp_grid.azimuth
        return super().__call__(
            theta, phi, dtheta=dtheta, dphi=dphi, grid=False)
