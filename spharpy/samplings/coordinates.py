import numpy as np
from pyfar.classes.coordinates import sph2cart, cyl2cart
import pyfar as pf


class SamplingSphere(pf.Coordinates):
    """Class for samplings on a sphere"""

    def __init__(
            self, x=None, y=None, z=None, n_max=None, weights: np.array = None,
            comment: str = "", radius_atol=1e-6):
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
        weights: array like, number, optional
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        sh_order : int, optional
            Maximum spherical harmonic order of the sampling grid.
            The default is ``None``.
        radius_atol : float, optional
            All points that are stored in a SamplingSphere must have the same
            radius and an error is raised if the maximum deviation from the
            mean radius exceeds this tolerance. The default of ``1e-6`` meter
            is intended to allow for some numerical inaccuracy.
        """
        # must be initialized first, because it is already required for
        # checking if input points have the same radius
        self._radius_atol = None
        self.radius_atol = radius_atol

        pf.Coordinates.__init__(
            self, x, y, z, weights=weights, comment=comment)
        self._n_max = n_max

    @classmethod
    def from_cartesian(
            cls, x, y, z, n_max=None, weights: np.array = None,
            comment: str = "", radius_atol: float = 1e-6):
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
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_atol : float, optional
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
            radius_atol=radius_atol)

    @classmethod
    def from_spherical_elevation(
            cls, azimuth, elevation, radius, n_max=None,
            weights: np.array = None, comment: str = "",
            radius_atol: float = 1e-6):
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
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_atol : float, optional
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
            radius_atol=radius_atol)

    @classmethod
    def from_spherical_colatitude(
            cls, azimuth, colatitude, radius, n_max=None,
            weights: np.array = None, comment: str = "",
            radius_atol: float = 1e-6):
        """Create a Coordinates class object from a set of points on a sphere.

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
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_atol : float, optional
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
            radius_atol=radius_atol)

    @classmethod
    def from_spherical_side(
            cls, lateral, polar, radius, n_max=None,
            weights: np.array = None, comment: str = "",
            radius_atol: float = 1e-6):
        """Create a Coordinates class object from a set of points on a sphere.

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
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_atol : float, optional
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
            radius_atol=radius_atol)

    @classmethod
    def from_spherical_front(
            cls, frontal, upper, radius, n_max=None, weights: np.array = None,
            comment: str = "", radius_atol: float = 1e-6):
        """Create a Coordinates class object from a set of points on a sphere.

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
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_atol : float, optional
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
            radius_atol=radius_atol)

    @classmethod
    def from_cylindrical(
            cls, azimuth, z, rho, n_max=None, weights: np.array = None,
            comment: str = "", radius_atol: float = 1e-6):
        """Create a Coordinates class object from a set of points on a sphere.

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
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        radius_atol : float, optional
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
            radius_atol=radius_atol)

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
    def radius_atol(self):
        """Get or set the radius tolerance in meter."""
        return self._radius_atol

    @radius_atol.setter
    def radius_atol(self, value):
        """Get or set the radius tolerance in meter."""
        if isinstance(value, (int, float)) and value > 0:
            self._radius_atol = float(value)
        else:
            raise ValueError(
                'The radius tolerance must be a number greater than zero')

    def _check_points(self, x, y, z):
        """Check input data before setting coordinates"""

        # convert to numpy arrays of the same shape
        x, y, z = super()._check_points(x, y, z)

        # check for equal radius
        radius = np.sqrt(x.flatten()**2 + y.flatten()**2 + z.flatten()**2)
        radius_delta = np.max(np.abs(np.mean(radius) - radius))
        if radius_delta > self.radius_atol:
            raise ValueError(
                'All points must have the same radius but the deviation from '
                f'the mean radius is {radius_delta:.3g} m, which exceeds the'
                f' tolerance of {self.radius_atol:.3g} m.')

        return x, y, z


