from scipy import interpolate as spinterpolate
import numpy as np
from spharpy._deprecation import convert_coordinates_to_pyfar


class SmoothSphereBivariateSpline(spinterpolate.SmoothSphereBivariateSpline):
    """Smooth bivariate spline approximation in spherical coordinates.
    The implementation uses the method proposed by Dierckx [#]_.

    Parameters
    ----------
    sampling : :class:`spharpy.samplings.Coordinates`, :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        Coordinates object containing the positions for which the data
        is sampled
    data : array, float
        Array containing the data at the sampling positions. Has to be
        real-valued.
    w : array, float
        Weighting coefficients
    s : float, 1e-4
        Smoothing factor > 0
        Positive smoothing factor defined for estimation condition:
        ``sum((w(i)*(r(i) - s(theta(i), phi(i))))**2, axis=0) <= s``
        Default ``s=len(w)`` which should be a good value if ``1/w[i]`` is
        an estimate of the standard deviation of ``r[i]``. The default
        value is ``1e-4``
    eps : float, 1e-16
        The eps valued to be considered for interpolator estimation.
        Depends on the used data type and numerical precision. The default
        is 1e-16.

    Note
    ----
    This is a wrapper for scipy's SmoothSphereBivariateSpline only using
    Coordinates objects as container for the azimuth and elevation angles.
    For detailed information see scipy's documentation.

    References
    ----------
    .. [#] P. Dierckx, “Algorithms for smoothing data on the sphere with
           tensor product splines,” p. 24, 1984.

    Examples
    --------
    >>> n_max = 10
    >>> sampling = spharpy.samplings.equalarea(
    ...     n_max, n_points=500, condition_num=np.inf)
    >>> Y = spharpy.spherical.spherical_harmonic_basis_real(n_max, sampling)
    >>> y_vec = spharpy.spherical.spherical_harmonic_basis_real(
    ...     n_max, Coordinates(1, 0, 0))
    >>> data = Y @ y_vec.T
    >>> data = np.sin(sampling.elevation)*np.sin(2*sampling.azimuth)
    >>> interp_grid = spharpy.samplings.hyperinterpolation(35)
    >>> data_grid = np.sin(interp_grid.elevation)*np.sin(2*interp_grid.azimuth)
    >>> interpolator = interpolate.SmoothSphereBivariateSpline(
    ...     sampling, data, s=1e-4)
    ...
    >>> interp_data = interpolator(interp_grid)

    """ # noqa: 501
    def __init__(self, sampling, data, w=None, s=1e-4, eps=1e-16):
        sampling = convert_coordinates_to_pyfar(sampling)
        theta = sampling.colatitude
        phi = sampling.azimuth
        if np.any(np.iscomplex(data)):
            raise ValueError("Complex data is not supported.")
        super().__init__(theta, phi, data, w, s, eps)

    def __call__(self, interp_grid, dtheta=0, dphi=0):
        """Evaluate the spline on a new sampling grid.

        Parameters
        ----------
        interp_grid : :class:`spharpy.samplings.Coordinates`, :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
            Coordinates object containing a new set of points for which data
            is to be interpolated.
        dtheta : int, optional
            Order of theta derivative
        dphi : int, optional
            Order of phi derivative

        """ # noqa: 501
        interp_grid = convert_coordinates_to_pyfar(interp_grid)
        theta = interp_grid.colatitude
        phi = interp_grid.azimuth
        return super().__call__(
            theta, phi, dtheta=dtheta, dphi=dphi, grid=False)

    def get_coeffs(self):
        return super().get_coeffs()

    def get_knots(self):
        return super().get_knots()

    def get_residual(self):
        return super().get_residual()
