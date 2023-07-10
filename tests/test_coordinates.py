import numpy as np
import numpy.testing as npt
import pytest

from spharpy.samplings import cart2latlon, cart2sph
from spharpy.samplings.coordinates import Coordinates, SamplingSphere

import pyfar as pf


def test_coordinates_init():
    coords = Coordinates()
    assert isinstance(coords, Coordinates)


def test_coordinates_init_val():

    coords = Coordinates(1, 0, 0)
    assert isinstance(coords, Coordinates)


def test_to_pyfar():
    coords = Coordinates(1, 0, 0)
    pyfar_coords = coords.to_pyfar()
    np.testing.assert_allclose(pyfar_coords.get_cart(), coords.cartesian.T)


def test_from_pyfar():
    pyfar_coords = pf.Coordinates(1, 0, 0)
    spharpy_coords = Coordinates.from_pyfar(pyfar_coords)
    np.testing.assert_allclose(
        pyfar_coords.get_cart(), spharpy_coords.cartesian.T)


def test_coordinates_init_incomplete():
    x = [1, 2]
    y = 1
    z = 1
    with pytest.raises(ValueError):
        Coordinates(x, y, z)
        pytest.fail("Input arrays need to have same dimensions.")


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

def test_coordinates_init_from_array_spherical():
    rad = [1., 1., 1., 1.]
    ele = [np.pi/2, np.pi/2, 0, np.pi/2]
    azi = [0, np.pi/2, 0, np.pi/4]

    points = np.array([rad, ele, azi])
    coords = Coordinates.from_array(points, coordinate_system='spherical')

    npt.assert_allclose(coords.radius, rad, atol=1e-15)
    npt.assert_allclose(coords.elevation, ele, atol=1e-15)
    npt.assert_allclose(coords.azimuth, azi, atol=1e-15)

def test_coordinates_init_from_array_cartesian():
    x = [1, 0, 0, 0]
    y = [0, 1, 0, 0]
    z = [0, 0, 1, 0]

    points = np.array([x, y, z])
    coords = Coordinates.from_array(points)

    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)

def test_getter_x():
    x = np.array([1, 0], dtype=float)
    coords = Coordinates()
    coords._x = x
    npt.assert_allclose(coords.x, x)

def test_getter_y():
    y = np.array([1, 0], dtype=float)
    coords = Coordinates()
    coords._y = y
    npt.assert_allclose(coords.y, y)

def test_getter_z():
    z = np.array([1, 0], dtype=float)
    coords = Coordinates()
    coords._z = z
    npt.assert_allclose(coords.z, z)

def test_setter_x():
    value = np.array([1.0, 1], dtype=float)
    coords = Coordinates()
    coords.x = value
    npt.assert_allclose(value, coords._x)

def test_setter_y():
    value = np.array([1.0, 1], dtype=float)
    coords = Coordinates()
    coords.y = value
    npt.assert_allclose(value, coords._y)

def test_setter_z():
    value = np.array([1.0, 1], dtype=float)
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

def test_getter_cartesian():
    x = [1, 0, 0, 0]
    y = [0, 1, 0, 0]
    z = [0, 0, 1, 0]

    coords = Coordinates(x, y, z)
    ref = np.vstack((x, y, z))
    npt.assert_allclose(coords.cartesian, ref)


def test_setter_cartesian():
    x = np.array([1, 0, 0, 0])
    y = np.array([0, 1, 0, 0])
    z = np.array([0, 0, 1, 0])
    cart = np.vstack((x, y, z))
    coords = Coordinates()
    coords.cartesian = cart
    npt.assert_allclose(coords.cartesian, cart)


def test_getter_spherical():
    x = np.array([1, 0, 0, 1], dtype=float)
    y = np.array([0, 1, 0, 1], dtype=float)
    z = np.array([0, 0, 1, 1], dtype=float)

    rad, theta, phi = cart2sph(x, y, z)

    coords = Coordinates(x, y, z)
    ref = np.vstack((rad, theta, phi))
    npt.assert_allclose(coords.spherical, ref)


def test_setter_spherical():
    eps = np.spacing(1)
    x = np.array([1, 0, 0, 1], dtype=float)
    y = np.array([0, 1, 0, 1], dtype=float)
    z = np.array([0, 0, 1, 1], dtype=float)
    rad, theta, phi = cart2sph(x, y, z)
    spherial = np.vstack((rad, theta, phi))
    coords = Coordinates()
    coords.spherical = spherial
    npt.assert_allclose(coords._x, x, atol=eps)
    npt.assert_allclose(coords._y, y, atol=eps)
    npt.assert_allclose(coords._z, z, atol=eps)


def test_n_points():
    coords = Coordinates([1, 0], [1, 1], [0, 1])
    assert coords.n_points == 2


def test_find_nearest():
    coords = Coordinates([1, 0], [1, 1], [0, 1])
    point = Coordinates(1, 1, 0)

    dist, idx = coords.find_nearest_point(point)

    assert idx == 0


def test_len():
    coords = Coordinates([1, 0], [1, 1], [0, 1])
    assert len(coords) == 2


def test_getitem():
    coords = Coordinates([1, 0], [1, 1], [0, 1])
    getcoords = coords[0]
    npt.assert_allclose(np.squeeze(getcoords.cartesian), np.array([1, 1, 0]))


def test_setitem():
    coords = Coordinates([0, 0], [1, 1], [0, 1])
    setcoords = Coordinates(1, 1, 0)
    coords[0] = setcoords
    npt.assert_allclose(np.squeeze(coords.cartesian),
                        np.array([[1, 0], [1, 1], [0, 1]]))


def test_sampling_sphere_init():
    sampling = SamplingSphere()
    assert isinstance(sampling, SamplingSphere)


def test_sampling_sphere_init_value():
    sampling = SamplingSphere(1, 0, 0, 0)
    assert isinstance(sampling, SamplingSphere)


def test_sampling_to_pyfar_coords():
    sampling = SamplingSphere(
        [1], [0], [0], n_max=0, weights=np.array([4*np.pi]))
    pyfar_coords = sampling.to_pyfar()
    np.testing.assert_allclose(pyfar_coords.get_cart(), sampling.cartesian.T)
    assert pyfar_coords.sh_order == sampling.n_max
    assert pyfar_coords.weights == 1.


def test_from_pyfar():
    pyfar_coords = pf.Coordinates(1, 0, 0, weights=1, sh_order=0)
    spharpy_sampling = SamplingSphere.from_pyfar(pyfar_coords)
    np.testing.assert_allclose(
        pyfar_coords.get_cart(), spharpy_sampling.cartesian.T)
    assert pyfar_coords.sh_order == spharpy_sampling.n_max
    npt.assert_almost_equal(4*np.pi, spharpy_sampling.weights)


def sampling_cube():
    """Helper function returning a cube sampling"""
    x = [1, -1, 0, 0, 0, 0]
    y = [0, 0, 1, -1, 0, 0]
    z = [0, 0, 0, 0, 1, -1]

    return x, y, z

def test_getter_n_max():
    x, y, z = sampling_cube()
    n_max = 1
    sampling = SamplingSphere(x, y, z, n_max)

    assert sampling.n_max == n_max

def test_setter_n_max():
    x, y, z = sampling_cube()
    n_max = 1
    sampling = SamplingSphere(x, y, z, 0)

    sampling.n_max = n_max
    assert sampling._n_max == n_max


def test_merge():
    s1 = Coordinates(1, 0, 0)
    s2 = Coordinates(0, 2, 0)

    s1.merge(s2)

    truth = np.array([[1, 0], [0, 2], [0, 0]])
    npt.assert_allclose(truth, s1.cartesian)
