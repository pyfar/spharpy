"""
The SamplingSphere class inherits from the :py:class:`pyfar.Coordinates` class
and is designed to represent a set of points on a sphere.

Therefore, all points must have the same radius within an absolute tolerance,
defined by :py:attr:`~spharpy.SamplingSphere.radius_tolerance`. If the
:py:attr:`~spharpy.SamplingSphere.weights` are not None, their sum must
equal the integral over the unit sphere, which is :math:`4\pi`.

It also adds two additional properties:

- :py:attr:`~spharpy.SamplingSphere.n_max`: the maximum spherical harmonic
  order of the sampling grid.
- :py:attr:`~spharpy.SamplingSphere.quadrature`: a flag that indicates if 
  the points belong to a quadrature, which requires that the
  :py:attr:`~spharpy.SamplingSphere.weights` are not None.

Note that the :py:mod:`spharpy.samplings` module provides a set of
predefined spherical sampling grids, which can be used to create a
:py:class:`spharpy.SamplingSphere` object.

"""

import numpy as np
from pyfar.classes.coordinates import sph2cart, cyl2cart
import pyfar as pf


class SamplingSphere(pf.Coordinates):
    """Class for samplings on a sphere"""

    def __init__(
            self, x=None, y=None, z=None, n_max=None, weights: np.array = None,
            quadrature: bool = False, comment: str = "",
            radius_tolerance=1e-6):
        r"""
        Create a SamplingSphere class object from a set of points on a sphere.

        See :py:mod:`pyfar.classes.coordinates` for more information.

        Parameters
        ----------
        x : ndarray, number
            X coordinate of a right handed Cartesian coordinate system in
            meters (-\infty < x < \infty).
        y : ndarray, number
            Y coordinate of a right handed Cartesian coordinate system in
            meters (-\infty < y < \infty).
        z : ndarray, number
            Z coordinate of a right handed Cartesian coordinate system in
            meters (-\infty < z < \infty).
        n_max : int, optional
            Maximum spherical harmonic order of the sampling grid.
            The default is ``None``.
        weights: array like, number, optional
            Weighting factors for coordinate points. Their sum must equal to
            the integral over the unit sphere, which is :math:`4\pi`.
            The `shape` of the array must match the `shape` of the individual
            coordinate arrays. The default is ``None``, which means that no
            weights are used.
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that `weights` is not ``None``. The default is ``False``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_tolerance : float, optional
            All points that are stored in a SamplingSphere must have the same
            radius and an error is raised if the maximum deviation from the
            mean radius exceeds this tolerance. The default of ``1e-6`` meter
            is intended to allow for some numerical inaccuracy.
        """
        self._radius_tolerance = None
        self.radius_tolerance = radius_tolerance

        pf.Coordinates.__init__(
            self, x, y, z, weights=weights, comment=comment)
        self._n_max = n_max

        # initialize and set quadrature
        self._quadrature = None
        self.quadrature = quadrature

    @classmethod
    def from_cartesian(
            cls, x, y, z, n_max=None, weights: np.array = None,
            quadrature: bool = False, comment: str = "",
            radius_tolerance: float = 1e-6):
        r"""
        Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`pyfar.classes.coordinates` for  more information.

        Parameters
        ----------
        x : ndarray, number
            X coordinate of a right handed Cartesian coordinate system in
            meters (-\infty < x < \infty).
        y : ndarray, number
            Y coordinate of a right handed Cartesian coordinate system in
            meters (-\infty < y < \infty).
        z : ndarray, number
            Z coordinate of a right handed Cartesian coordinate system in
            meters (-\infty < z < \infty).
        n_max : int, optional
            Maximum spherical harmonic order of the sampling grid.
            The default is ``None``.
        weights: array like, number, optional
            Weighting factors for coordinate points. Their sum must equal to
            the integral over the unit sphere, which is :math:`4\pi`.
            The `shape` of the array must match the `shape` of the individual
            coordinate arrays. The default is ``None``, which means that no
            weights are used.
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that `weights` is not ``None``. The default is ``False``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_tolerance : float, optional
            All points that are stored in a SamplingSphere must have the same
            radius and an error is raised if the maximum deviation from the
            mean radius exceeds this tolerance. The default of ``1e-6`` meter
            is intended to allow for some numerical inaccuracy.

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_cartesian(0, 0, 1)
        Or the using init
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere(0, 0, 1)
        """
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max,
            quadrature=quadrature, radius_tolerance=radius_tolerance)

    @classmethod
    def from_spherical_elevation(
            cls, azimuth, elevation, radius, n_max=None,
            weights: np.array = None, quadrature: bool = False,
            comment: str = "", radius_tolerance: float = 1e-6):
        """Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`pyfar.classes.coordinates` for  more information.

        Parameters
        ----------
        azimuth : ndarray, double
            Angle in radiant of rotation from the x-y-plane facing towards
            positive x direction. Used for spherical and cylindrical coordinate
            systems.
        elevation : ndarray, double
            Angle in radiant with respect to horizontal plane (x-z-plane).
            Used for spherical coordinate systems.
        radius : ndarray, double
            Distance to origin for each point. Used for spherical coordinate
            systems.
        n_max : int, optional
            Maximum spherical harmonic order of the sampling grid.
            The default is ``None``.
        weights: array like, float, None, optional
            Weighting factors for coordinate points. Their sum must equal to
            the integral over the unit sphere, which is :math:`4\pi`.
            The `shape` of the array must match the `shape` of the individual
            coordinate arrays. The default is ``None``, which means that no
            weights are used.
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that `weights` is not ``None``. The default is ``False``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_tolerance : float, optional
            All points that are stored in a SamplingSphere must have the same
            radius and an error is raised if the maximum deviation from the
            mean radius exceeds this tolerance. The default of ``1e-6`` meter
            is intended to allow for some numerical inaccuracy.

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_spherical_elevation(0, 0, 1)
        """

        x, y, z = sph2cart(
            azimuth, np.pi / 2 - np.atleast_1d(elevation), radius)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max,
            quadrature=quadrature, radius_tolerance=radius_tolerance)

    @classmethod
    def from_spherical_colatitude(
            cls, azimuth, colatitude, radius, n_max=None,
            weights: np.array = None, quadrature: bool = False,
            comment: str = "", radius_tolerance: float = 1e-6):
        r"""Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`pyfar.classes.coordinates` for  more information.

        Parameters
        ----------
        azimuth : ndarray, double
            Angle in radiant of rotation from the x-y-plane facing towards
            positive x direction. Used for spherical and cylindrical coordinate
            systems.
        colatitude : ndarray, double
            Angle in radiant with respect to polar axis (z-axis). Used for
            spherical coordinate systems.
        radius : ndarray, double
            Distance to origin for each point. Used for spherical coordinate
            systems.
        n_max : int, optional
            Maximum spherical harmonic order of the sampling grid.
            The default is ``None``.
        weights: array like, number, optional
            Weighting factors for coordinate points. Their sum must equal to
            the integral over the unit sphere, which is :math:`4\pi`.
            The `shape` of the array must match the `shape` of the individual
            coordinate arrays. The default is ``None``, which means that no
            weights are used.
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that `weights` is not ``None``. The default is ``False``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_tolerance : float, optional
            All points that are stored in a SamplingSphere must have the same
            radius and an error is raised if the maximum deviation from the
            mean radius exceeds this tolerance. The default of ``1e-6`` meter
            is intended to allow for some numerical inaccuracy.

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_spherical_colatitude(0, 0, 1)
        """

        x, y, z = sph2cart(azimuth, colatitude, radius)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max,
            quadrature=quadrature, radius_tolerance=radius_tolerance)

    @classmethod
    def from_spherical_side(
            cls, lateral, polar, radius, n_max=None,
            weights: np.array = None, quadrature: bool = False,
            comment: str = "", radius_tolerance: float = 1e-6):
        r"""Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`pyfar.classes.coordinates` for  more information.

        Parameters
        ----------
        lateral : ndarray, double
            Angle in radiant with respect to horizontal plane (x-y-plane).
            Used for spherical coordinate systems.
        polar : ndarray, double
            Angle in radiant of rotation from the x-z-plane facing towards
            positive x direction. Used for spherical coordinate systems.
        radius : ndarray, double
            Distance to origin for each point. Used for spherical coordinate
            systems.
        n_max : int, optional
            Maximum spherical harmonic order of the sampling grid.
            The default is ``None``.
        weights: array like, number, optional
            Weighting factors for coordinate points. Their sum must equal to
            the integral over the unit sphere, which is :math:`4\pi`.
            The `shape` of the array must match the `shape` of the individual
            coordinate arrays. The default is ``None``, which means that no
            weights are used.
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that `weights` is not ``None``. The default is ``False``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_tolerance : float, optional
            All points that are stored in a SamplingSphere must have the same
            radius and an error is raised if the maximum deviation from the
            mean radius exceeds this tolerance. The default of ``1e-6`` meter
            is intended to allow for some numerical inaccuracy.

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_spherical_side(0, 0, 1)
        """

        x, z, y = sph2cart(
            polar, np.pi / 2 - np.atleast_1d(lateral), radius)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max,
            quadrature=quadrature, radius_tolerance=radius_tolerance)

    @classmethod
    def from_spherical_front(
            cls, frontal, upper, radius, n_max=None, weights: np.array = None,
            quadrature: bool = False, comment: str = "",
            radius_tolerance: float = 1e-6):
        r"""Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`pyfar.classes.coordinates` for  more information.

        Parameters
        ----------
        frontal : ndarray, double
            Angle in radiant of rotation from the y-z-plane facing towards
            positive y direction. Used for spherical coordinate systems.
        upper : ndarray, double
            Angle in radiant with respect to polar axis (x-axis). Used for
            spherical coordinate systems.
        radius : ndarray, double
            Distance to origin for each point. Used for spherical coordinate
            systems.
        n_max : int, optional
            Maximum spherical harmonic order of the sampling grid.
            The default is ``None``.
        weights: array like, number, optional
            Weighting factors for coordinate points. Their sum must equal to
            the integral over the unit sphere, which is :math:`4\pi`.
            The `shape` of the array must match the `shape` of the individual
            coordinate arrays. The default is ``None``, which means that no
            weights are used.
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that `weights` is not ``None``. The default is ``False``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_tolerance : float, optional
            All points that are stored in a SamplingSphere must have the same
            radius and an error is raised if the maximum deviation from the
            mean radius exceeds this tolerance. The default of ``1e-6`` meter
            is intended to allow for some numerical inaccuracy.

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_spherical_front(0, 0, 1)
        """

        y, z, x = sph2cart(frontal, upper, radius)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max,
            quadrature=quadrature, radius_tolerance=radius_tolerance)

    @classmethod
    def from_cylindrical(
            cls, azimuth, z, rho, n_max=None, weights: np.array = None,
            quadrature: bool = False, comment: str = "",
            radius_tolerance: float = 1e-6):
        r"""Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`pyfar.classes.coordinates` for  more information.

        Parameters
        ----------
        azimuth : ndarray, double
            Angle in radiant of rotation from the x-y-plane facing towards
            positive x direction. Used for spherical and cylindrical coordinate
            systems.
        z : ndarray, double
            The z coordinate
        rho : ndarray, double
            Distance to origin for each point in the x-y-plane. Used for
            cylindrical coordinate systems.
        n_max : int, optional
            Maximum spherical harmonic order of the sampling grid.
            The default is ``None``.
        weights: array like, number, optional
            Weighting factors for coordinate points. Their sum must equal to
            the integral over the unit sphere, which is :math:`4\pi`.
            The `shape` of the array must match the `shape` of the individual
            coordinate arrays. The default is ``None``, which means that no
            weights are used.
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that `weights` is not ``None``. The default is ``False``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_tolerance : float, optional
            All points that are stored in a SamplingSphere must have the same
            radius and an error is raised if the maximum deviation from the
            mean radius exceeds this tolerance. The default of ``1e-6`` meter
            is intended to allow for some numerical inaccuracy.

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_cylindrical(0, 0, 1, sh_order=1)
        """

        x, y, z = cyl2cart(azimuth, z, rho)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max,
            quadrature=quadrature, radius_tolerance=radius_tolerance)

    @property
    def n_max(self):
        """Get the maximum spherical harmonic order."""
        return self._n_max

    @n_max.setter
    def n_max(self, value):
        """Set the maximum spherical harmonic order."""
        assert value >= 0
        if value is None:
            self._n_max = None
        else:
            self._n_max = int(value)

    @property
    def radius_tolerance(self):
        """Get or set the radius tolerance in meter."""
        return self._radius_tolerance

    @radius_tolerance.setter
    def radius_tolerance(self, value):
        """Get or set the radius tolerance in meter."""

        # check input
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(
                'The radius tolerance must be a number greater than zero')

        current_tolerance = self.radius_tolerance
        self._radius_tolerance = float(value)

        # Check if points meet new tolerance if points exist
        if hasattr(self, 'x'):
            try:
                self._check_points(self._x, self._y, self._z)
            except ValueError as e:
                # revert setting the tolerance and raise the error
                self._radius_tolerance = current_tolerance
                raise e

    def _check_points(self, x, y, z):
        """Check input data before setting coordinates"""

        # convert to numpy arrays of the same shape
        x, y, z = super()._check_points(x, y, z)

        # check for equal radius
        radius = np.sqrt(x.flatten()**2 + y.flatten()**2 + z.flatten()**2)
        radius_delta = np.max(radius) - np.min(radius)
        if radius_delta > self.radius_tolerance:
            raise ValueError(
                'All points must have the same radius but the difference '
                f'between the minimum and maximum radius is {radius_delta:.3g}'
                ' m, which exceeds the tolerance of '
                f'{self.radius_tolerance:.3g} m. The tolerance can be changed '
                'using SamplingSphere.radius_tolerance.')

        return x, y, z


    def _check_weights(self, weights):
        r"""Check if the weights are valid.
        The weights must be positive and their sum must equal integration of
        the unit sphere, i.e. :math:`4\pi`.

        Parameters
        ----------
        weights : array like, number
            the weights for each point, should be of size of self.csize.

        Returns
        -------
        weights : np.ndarray[float64], None
            The weights reshaped to the cshape of the coordinates if not None.
            Otherwise None.
        """
        weights = super()._check_weights(weights)

        if weights is None:
            return weights
        if np.any(weights < 0) or np.any(np.isnan(weights)):
            raise ValueError("All weights must be positive numeric values.")

        if not np.isclose(np.sum(weights), 4*np.pi, atol=1e-6, rtol=1e-6):
            raise ValueError(
                "The sum of the weights must be equal to 4*pi. "
                f"Current sum: {np.sum(weights)}")

        return weights

    @property
    def weights(self):
        r"""The area/quadrature weights of the sampling.
        Their sum must equal to :math:`4\pi`.
        """
        return super().weights

    @weights.setter
    def weights(self, weights):
        r"""The area/quadrature weights of the sampling.
        Their sum must equal to :math:`4\pi`.
        """
        super(__class__, type(self)).weights.fset(self, weights)

    @property
    def quadrature(self):
        """Get or set the quadrature flag."""
        return self._quadrature

    @quadrature.setter
    def quadrature(self, value):
        """Get or set the quadrature flag."""

        # check input
        if not isinstance(value, bool):
            raise TypeError(
                f'quadrature must be True or False but is {value}')

        # check if weights are set
        # (if they are the weights setter enforces that they sum to 4 pi)
        if self.weights is None and value:
            raise ValueError(
                'quadrature can not be True because the weights are None')

        self._quadrature = value
