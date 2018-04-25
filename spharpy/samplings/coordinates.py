import numpy as np
from spharpy.samplings.helpers import sph2cart

class Coordinates(object):
    """Container class for coordinates in a three-dimensional space, allowing
    for compact representation and convenient conversion into spherical as well
    as geospatial coordinate systems.
    The constructor as well as the internal representation are only
    available in Cartesian coordinates. To create a Coordinates object from
    a set of points in spherical coordinates, please use the
    Coordinates.from_spherical() method.

    Attributes
    ----------
    x : ndarray, double
        x-coordinate
    y : ndarray, double
        y-coordinate
    z : ndarray, double
        z-coordinate

    """
    def __init__(self, x=None, y=None, z=None):
        """Init coordinates container

        Attributes
        ----------
        x : ndarray, double
            x-coordinate
        y : ndarray, double
            y-coordinate
        z : ndarray, double
            z-coordinate
        """

        super(Coordinates, self).__init__()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

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
        self._x = np.asarray(value, dtype=np.float64)

    @property
    def y(self):
        """The y-axis coordinate for each point."""
        return self._y

    @y.setter
    def y(self, value):
        self._y = np.asarray(value, dtype=np.float64)

    @property
    def z(self):
        """The z-axis coordinate for each point."""
        return self._z

    @z.setter
    def z(self, value):
        self._z = np.asarray(value, dtype=np.float64)

    @property
    def radius(self):
        """The radius for each point."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @radius.setter
    def radius(self, radius):
        x, y, z = sph2cart(np.asarray(radius, dtype=np.float64),
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
        x, y, z = sph2cart(self.radius,
                           self.elevation,
                           np.asarray(azimuth, dtype=np.float64))
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
        x, y, z = sph2cart(self.radius,
                           np.asarray(elevation, dtype=np.float64),
                           self.azimuth)
        self._x = x
        self._y = y
        self._z = z

    @classmethod
    def from_cartesian(cls, x, y, z):
        """Create a Coordinates class object from a set of points in the
        Cartesian coordinate system.

        Parameters
        ----------
        x : ndarray, double
            x-coordinate
        y : ndarray, double
            y-coordinate
        z : ndarray, double
            z-coordinate
        """
        return Coordinates(x, y, z)

    @classmethod
    def from_spherical(cls, radius, elevation, azimuth):
        """Create a Coordinates class object from a set of points in the
        spherical coordinate system.

        Parameters
        ----------
        radius : ndarray, double
            The radius for each point
        elevation : ndarray, double
            The elevation angle in radians
        azimuth : ndarray, double
            The azimuth angle in radians
        """
        radius = np.asarray(radius, dtype=np.double)
        elevation = np.asarray(elevation, dtype=np.double)
        azimuth = np.asarray(azimuth, dtype=np.double)
        x, y, z = sph2cart(radius, elevation, azimuth)
        return Coordinates(x, y, z)

    @property
    def latitude(self):
        """The latitude angle as used in geospatial coordinates."""
        return np.pi/2 - self.elevation

    @property
    def longitude(self):
        """The longitude angle as used in geospatial coordinates."""
        return np.arctan2(self.y, self.x)

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
        x, y, z = sph2cart(value[0, :], value[1, :], value[2, :])
        self.cartesian = np.vstack((x, y, z))
