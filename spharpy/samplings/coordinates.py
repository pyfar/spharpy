import numpy as np
from pyfar.classes.coordinates import sph2cart, cyl2cart
import pyfar as pf


class SamplingSphere(pf.Coordinates):
    """Class for samplings on a sphere"""

    def __init__(
            self, x=None, y=None, z=None, n_max=None, weights: np.array = None,
            quadrature: bool = False, comment: str = ""):
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
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that all `weights` are greater than zero and sum to
            :math:`4\pi`. The default is ``False``.
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

        # initialize and set quadrature
        self._quadrature = None
        self.quadrature = quadrature

    @classmethod
    def from_cartesian(
            cls, x, y, z, n_max=None, weights: np.array = None,
            quadrature: bool = False, comment: str = ""):
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
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that all `weights` are greater than zero and sum to
            :math:`4\pi`. The default is ``False``.
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
            x, y, z, weights=weights, comment=comment, n_max=n_max,
            quadrature=quadrature)

    @classmethod
    def from_spherical_elevation(
            cls, azimuth, elevation, radius, n_max=None,
            weights: np.array = None, quadrature: bool = False,
            comment: str = ""):
        r"""Create a Coordinates class object from a set of points on a sphere.

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
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that all `weights` are greater than zero and sum to
            :math:`4\pi`. The default is ``False``.
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
        return cls(x, y, z, weights=weights, comment=comment, n_max=n_max,
                   quadrature=quadrature)

    @classmethod
    def from_spherical_colatitude(
            cls, azimuth, colatitude, radius, n_max=None,
            weights: np.array = None, quadrature: bool = False,
            comment: str = ""):
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
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that all `weights` are greater than zero and sum to
            :math:`4\pi`. The default is ``False``.
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
        return cls(x, y, z, weights=weights, comment=comment, n_max=n_max,
                   quadrature=quadrature)

    @classmethod
    def from_spherical_side(
            cls, lateral, polar, radius, n_max=None,
            weights: np.array = None, quadrature: bool = False,
            comment: str = ""):
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
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that all `weights` are greater than zero and sum to
            :math:`4\pi`. The default is ``False``.
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
        return cls(x, y, z, weights=weights, comment=comment, n_max=n_max,
                   quadrature=quadrature)

    @classmethod
    def from_spherical_front(
            cls, frontal, upper, radius, n_max=None, weights: np.array = None,
            quadrature: bool = False, comment: str = ""):
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
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that all `weights` are greater than zero and sum to
            :math:`4\pi`. The default is ``False``.
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
        return cls(x, y, z, weights=weights, comment=comment, n_max=n_max,
                   quadrature=quadrature)

    @classmethod
    def from_cylindrical(
            cls, azimuth, z, rho, n_max=None, weights: np.array = None,
            quadrature: bool = False, comment: str = ""):
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
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        quadrature : bool, optional
            Flag that indicates if points belong to a quadrature, which
            requires that all `weights` are greater than zero and sum to
            :math:`4\pi`. The default is ``False``.
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
        return cls(x, y, z, weights=weights, comment=comment, n_max=n_max,
                   quadrature=quadrature)

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

        # flag indicating if weights sum to 4 pi
        if self.weights is None and value:
            raise ValueError(
                'quadrature can not be True because the weights are None')

        self._quadrature = value

