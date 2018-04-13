import numpy as np
import numpy.testing as npt
from pytest import raises

from spharpy.samplings.coordinates import Coordinates
from spharpy.samplings import sph2cart, cart2sph, cart2latlon

def test_coordinates_init():
    coords = Coordinates()
    assert isinstance(coords, Coordinates)

def test_coordinates_init_val():

    coords = Coordinates(1, 0, 0)
    assert isinstance(coords, Coordinates)

def test_coordinates_init_incomplete():
    x = [1, 2]
    y = 1
    z = 1
    with raises(ValueError, \
            message="Input arrays need to have same dimensions."):
        Coordinates(x, y, z)

def test_coordinates_init_from_cartesian():
    x = 1
    y = 0
    z = 0
    coords = Coordinates.from_cartesian(x, y, z)
    npt.assert_allclose(coords._x, x)
    npt.assert_allclose(coords._y, y)
    npt.assert_allclose(coords._z, z)


def test_coordinates_init_from_spherical():
    x = 1
    y = 0
    z = 0
    rad, theta, phi = cart2sph(x, y, z)
    coords = Coordinates.from_spherical(rad, theta, phi)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)

def test_getter_x():
    x = np.array([1, 0], dtype=np.double)
    coords = Coordinates()
    coords._x = x
    npt.assert_allclose(coords.x, x)

def test_getter_y():
    y = np.array([1, 0], dtype=np.double)
    coords = Coordinates()
    coords._y = y
    npt.assert_allclose(coords.y, y)

def test_getter_z():
    z = np.array([1, 0], dtype=np.double)
    coords = Coordinates()
    coords._z = z
    npt.assert_allclose(coords.z, z)

def test_setter_x():
    value = np.array([1.0, 1], dtype=np.double)
    coords = Coordinates()
    coords.x = value
    npt.assert_allclose(value, coords._x)

def test_setter_y():
    value = np.array([1.0, 1], dtype=np.double)
    coords = Coordinates()
    coords.y = value
    npt.assert_allclose(value, coords._y)

def test_setter_z():
    value = np.array([1.0, 1], dtype=np.double)
    coords = Coordinates()
    coords.z = value
    npt.assert_allclose(value, coords._z)

def test_getter_ele():
    value = np.pi/2
    coords = Coordinates()
    coords.z = 0
    coords.y = 0
    coords.x = 1
    npt.assert_allclose(coords.elevation, value)

def test_getter_radius():
    value = 1
    coords = Coordinates()
    coords.z = 0
    coords.y = 1
    coords.x = 0
    npt.assert_allclose(coords.radius, value)

def test_getter_azi():
    azi = np.pi/2
    coords = Coordinates()
    coords.z = 0
    coords.y = 1
    coords.x = 0
    npt.assert_allclose(coords.azimuth, azi)

def test_setter_rad():
    eps = np.spacing(1)
    rad = 0.5
    x = 0.5
    y = 0
    z = 0
    coords = Coordinates(1, 0, 0)
    coords.radius = rad
    npt.assert_allclose(coords._x, x, atol=eps)
    npt.assert_allclose(coords._y, y, atol=eps)
    npt.assert_allclose(coords._z, z, atol=eps)

def test_setter_ele():
    eps = np.spacing(1)
    ele = 0
    x = 0
    y = 0
    z = 1
    coords = Coordinates(1, 0, 0)
    coords.elevation = ele
    npt.assert_allclose(coords._x, x, atol=eps)
    npt.assert_allclose(coords._y, y, atol=eps)
    npt.assert_allclose(coords._z, z, atol=eps)


def test_setter_azi():
    eps = np.spacing(1)
    azi = np.pi/2
    x = 0
    y = 1
    z = 0
    coords = Coordinates(1, 0, 0)
    coords.azimuth = azi
    npt.assert_allclose(coords._x, x, atol=eps)
    npt.assert_allclose(coords._y, y, atol=eps)
    npt.assert_allclose(coords._z, z, atol=eps)

def test_getter_latitude():
    x = 1
    y = 0
    z = 0.5

    height, lat, lon = cart2latlon(x, y, z)
    coords = Coordinates(x, y, z)
    npt.assert_allclose(coords.latitude, lat)

def test_getter_longitude():
    x = 1
    y = 0
    z = 0.5

    height, lat, lon = cart2latlon(x, y, z)
    coords = Coordinates(x, y, z)
    npt.assert_allclose(coords.longitude, lon)
