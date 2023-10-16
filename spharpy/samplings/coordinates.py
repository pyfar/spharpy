import numpy as np
from pyfar.classes.coordinates import sph2cart, cyl2cart
import pyfar as pf
from spharpy.samplings.helpers import sph2cart as sp_sph2cart
from scipy.spatial import cKDTree


class SamplingSphere(pf.Coordinates):
    """Class for samplings on a sphere"""

    def __init__(
            self, x=None, y=None, z=None, n_max=None, weights: np.array = None,
            comment: str = ""):
        r"""
        Create a SamplingSphere class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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
        """
        pf.Coordinates.__init__(
            self, x, y, z, weights=weights, comment=comment)
        self._n_max = n_max

        super(Coordinates, self).__init__()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        if not np.shape(x) == np.shape(y) == np.shape(z):
            raise ValueError("Input arrays need to have same dimensions.")

        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        """The x-axis coordinates for each point.
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.asarray(value, dtype=float)

    @property
    def y(self):
        """The y-axis coordinate for each point."""
        return self._y

    @y.setter
    def y(self, value):
        self._y = np.asarray(value, dtype=float)

    @property
    def z(self):
        """The z-axis coordinate for each point."""
        return self._z

    @z.setter
    def z(self, value):
        self._z = np.asarray(value, dtype=float)

    @property
    def radius(self):
        """The radius for each point."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @radius.setter
    def radius(self, radius):
        x, y, z = sp_sph2cart(
            np.asarray(radius, dtype=float),
            self.elevation,
            self.azimuth)
        self._x = x
        self._y = y
        self._z = z

    @property
    def azimuth(self):
        """The azimuth angle for each point."""
        return np.mod(np.arctan2(self.y, self.x), 2*np.pi)

    @azimuth.setter
    def azimuth(self, azimuth):
        x, y, z = sp_sph2cart(
            self.radius,
            self.elevation,
            np.asarray(azimuth, dtype=float))
        self._x = x
        self._y = y
        self._z = z

    @property
    def elevation(self):
        """The elevation angle for each point"""
        rad = self.radius
        return np.arccos(self.z/rad)

    @elevation.setter
    def elevation(self, elevation):
        x, y, z = sp_sph2cart(
            self.radius,
            np.asarray(elevation, dtype=float),
            self.azimuth)
        self._x = x
        self._y = y
        self._z = z

    @classmethod
    def from_cartesian(
            cls, x, y, z, n_max=None, weights: np.array = None,
            comment: str = ""):
        r"""
        Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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
            x, y, z, weights=weights, comment=comment, n_max=n_max)

    @classmethod
    def from_spherical_elevation(
            cls, azimuth, elevation, radius, n_max=None,
            weights: np.array = None, comment: str = ""):
        """Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_spherical_elevation(0, 0, 1)
        """
        radius = np.asarray(radius, dtype=float)
        elevation = np.asarray(elevation, dtype=float)
        azimuth = np.asarray(azimuth, dtype=float)
        x, y, z = sp_sph2cart(radius, elevation, azimuth)
        return Coordinates(x, y, z)

        x, y, z = sph2cart(azimuth, np.pi / 2 - elevation, radius)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max)

    @classmethod
    def from_spherical_colatitude(
            cls, azimuth, colatitude, radius, n_max=None,
            weights: np.array = None, comment: str = ""):
        """Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_spherical_colatitude(0, 0, 1)
        """

        x, y, z = sph2cart(azimuth, colatitude, radius)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max)

    @classmethod
    def from_spherical_side(
            cls, lateral, polar, radius, n_max=None,
            weights: np.array = None, comment: str = ""):
        """Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

    @property
    def cartesian(self):
        """Cartesian coordinates of all points."""
        return np.vstack((self.x, self.y, self.z))

    @cartesian.setter
    def cartesian(self, value):
        """Cartesian coordinates of all points."""
        self.x = value[0, :]
        self.y = value[1, :]
        self.z = value[2, :]

    @property
    def spherical(self):
        """Spherical coordinates of all points."""
        return np.vstack((self.radius, self.elevation, self.azimuth))

    @spherical.setter
    def spherical(self, value):
        """Cartesian coordinates of all points."""
        x, y, z = sp_sph2cart(value[0, :], value[1, :], value[2, :])
        self.cartesian = np.vstack((x, y, z))

    @property
    def n_points(self):
        """Return number of points stored in the object"""
        return self.x.size

    def merge(self, other):
        """Merge another coordinates objects into this object."""
        data = np.concatenate(
            (self.cartesian, other.cartesian),
            axis=-1
        )
        self.cartesian = data

    def find_nearest_point(self, point):
        """Find the closest Coordinate point to a given Point.
        The search for the nearest point is performed using the scipy
        cKDTree implementation.

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

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_spherical_side(0, 0, 1)
        """

        x, z, y = sph2cart(polar, np.pi / 2 - lateral, radius)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max)

    @classmethod
    def from_spherical_front(
            cls, frontal, upper, radius, n_max=None, weights: np.array = None,
            comment: str = ""):
        """Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_spherical_front(0, 0, 1)
        """

        y, z, x = sph2cart(frontal, upper, radius)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max)

    def to_pyfar(self):
        """Export to a pyfar Coordinates object.

        Returns
        -------
        :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
            The equivalent pyfar class object.
        """
        return pf.Coordinates(
            self.x,
            self.y,
            self.z,
            domain='cart',
            convention='right',
            unit='met')

    @classmethod
    def from_cylindrical(
            cls, azimuth, z, rho, n_max=None, weights: np.array = None,
            comment: str = ""):
        """Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_cylindrical(0, 0, 1, sh_order=1)
        """

        x, y, z = cyl2cart(azimuth, z, rho)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max)

class SamplingSphere(pf.Coordinates):
    """Class for samplings on a sphere"""

    def __init__(
            self, x=None, y=None, z=None, n_max=None, weights: np.array = None,
            comment: str = ""):
        r"""
        Create a SamplingSphere class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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
        """
        pf.Coordinates.__init__(
            self, x, y, z, weights=weights, comment=comment)
        self._n_max = n_max

    @classmethod
    def from_cartesian(
            cls, x, y, z, n_max=None, weights: np.array = None,
            comment: str = ""):
        r"""
        Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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
            x, y, z, weights=weights, comment=comment, n_max=n_max)

    @classmethod
    def from_spherical_elevation(
            cls, azimuth, elevation, radius, n_max=None,
            weights: np.array = None, comment: str = ""):
        """Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_spherical_elevation(0, 0, 1)
        """

        x, y, z = sph2cart(azimuth, np.pi / 2 - elevation, radius)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max)

    @classmethod
    def from_spherical_colatitude(
            cls, azimuth, colatitude, radius, n_max=None,
            weights: np.array = None, comment: str = ""):
        """Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_spherical_colatitude(0, 0, 1)
        """

        x, y, z = sph2cart(azimuth, colatitude, radius)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max)

    @classmethod
    def from_spherical_side(
            cls, lateral, polar, radius, n_max=None,
            weights: np.array = None, comment: str = ""):
        """Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_spherical_side(0, 0, 1)
        """

        x, z, y = sph2cart(polar, np.pi / 2 - lateral, radius)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max)

    @classmethod
    def from_spherical_front(
            cls, frontal, upper, radius, n_max=None, weights: np.array = None,
            comment: str = ""):
        """Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_spherical_front(0, 0, 1)
        """

        y, z, x = sph2cart(frontal, upper, radius)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max)

    @classmethod
    def from_cylindrical(
            cls, azimuth, z, rho, n_max=None, weights: np.array = None,
            comment: str = ""):
        """Create a Coordinates class object from a set of points on a sphere.

        See :py:mod:`coordinates concepts <pyfar._concepts.coordinates>` for
        more information.

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

        Examples
        --------
        Create a SamplingSphere object
        >>> import pyfar as pf
        >>> sampling = pf.SamplingSphere.from_cylindrical(0, 0, 1, sh_order=1)
        """

        x, y, z = cyl2cart(azimuth, z, rho)
        return cls(
            x, y, z, weights=weights, comment=comment, n_max=n_max)

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
