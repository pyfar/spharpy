import numpy as np
import pytest

import spharpy.samplings as samplings
from spharpy.samplings.coordinates import SamplingSphere
from pyfar import Coordinates
import numpy.testing as npt
from pytest import raises


def test_cube_equidistant():
    n_points = 3
    coords = samplings.cube_equidistant(n_points)
    x = np.tile(np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1]), 3)
    y = np.hstack((np.ones(9) * -1, np.zeros(9), np.ones(9)))
    z = np.tile(np.array([-1, 0, 1]), 9)
    np.testing.assert_allclose(x, coords.x)
    np.testing.assert_allclose(y, coords.y)
    np.testing.assert_allclose(z, coords.z)


def test_cube_equidistant_pyfar():
    # test with int
    c = samplings.cube_equidistant(3)
    assert isinstance(c, Coordinates)
    assert c.csize == 3**3

    # test with tuple
    c = samplings.cube_equidistant((3, 2, 4))
    assert c.csize == 3*2*4


def test_hyperinterpolation():
    n_max = 1
    samplings.samplings._sph_extremal_load_data(n_max)
    sampling = samplings.hyperinterpolation(n_max=n_max)
    assert sampling.radius.size == (n_max+1)**2


def test_sph_extremal():
    # load test data
    samplings.samplings._sph_extremal_load_data([1, 10])

    # test without parameters
    assert samplings.hyperinterpolation() is None

    # test with n_points
    c = samplings.hyperinterpolation(4)
    isinstance(c, Coordinates)
    assert c.csize == 4
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with spherical harmonic order
    c = samplings.hyperinterpolation(n_max=1)
    assert c.csize == 4
    npt.assert_allclose(np.sum(c.weights), 1)

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.hyperinterpolation(4, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)

    # test loading SH order > 9
    c = samplings.hyperinterpolation(n_max=10)

    # test exceptions
    with raises(ValueError):
        c = samplings.hyperinterpolation(4, 1)
    with raises(ValueError):
        c = samplings.hyperinterpolation(5)
    with raises(ValueError):
        c = samplings.hyperinterpolation(n_max=0)


def test_spherical_t_design_const_e():
    order = 2
    samplings.samplings._sph_t_design_load_data(range(1, 11))
    coords = samplings.spherical_t_design(
        n_max=order, criterion='const_energy')
    assert isinstance(coords, SamplingSphere)


def test_spherical_t_design_const_angle():
    order = 2
    samplings.samplings._sph_t_design_load_data(range(1, 11))
    coords = samplings.spherical_t_design(
        n_max=order, criterion='const_angular_spread')
    assert isinstance(coords, SamplingSphere)


def test_spherical_t_design_invalid():
    order = 2
    samplings.samplings._sph_t_design_load_data(range(1, 11))
    with pytest.raises(ValueError, match='Invalid design'):
        samplings.spherical_t_design(n_max=order, criterion='bla')


def test_sph_t_design():
    # load test data
    samplings.samplings._sph_t_design_load_data(np.arange(1, 11))

    # test without parameters
    assert samplings.spherical_t_design() is None

    # test with degree
    c = samplings.spherical_t_design(2)
    isinstance(c, Coordinates)
    assert c.csize == 6

    # test with spherical harmonic order
    c = samplings.spherical_t_design(n_max=1)
    assert c.csize == 6
    c = samplings.spherical_t_design(
        n_max=1, criterion='const_angular_spread')
    assert c.csize == 8

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.spherical_t_design(2, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)

    # test loading degree order > 9
    c = samplings.spherical_t_design(10)

    # test exceptions
    with raises(ValueError):
        c = samplings.spherical_t_design(4, 1)
    with raises(ValueError):
        c = samplings.spherical_t_design(0)
    with raises(ValueError):
        c = samplings.spherical_t_design(n_max=0)
    with raises(ValueError):
        c = samplings.spherical_t_design(2, criterion='const_thread')


def test_dodecahedron():
    sampling = samplings.dodecahedron()
    assert isinstance(sampling, SamplingSphere)


def test_sph_dodecahedron():
    # test with default radius
    c = samplings.dodecahedron()
    assert isinstance(c, Coordinates)
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test with user radius
    c = samplings.dodecahedron(1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)


def test_icosahedron():
    sampling = samplings.icosahedron()
    assert isinstance(sampling, SamplingSphere)


def test_sph_icosahedron():
    # test with default radius
    c = samplings.icosahedron()
    assert isinstance(c, Coordinates)
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test with user radius
    c = samplings.icosahedron(1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)


def test_equiangular():
    n_max = 1
    sampling = samplings.equiangular(n_max=n_max)
    assert isinstance(sampling, SamplingSphere)


def test_equiangular_pyfar():
    # test without parameters
    with raises(ValueError):
        samplings.equiangular()

    # test with single number of points
    c = samplings.equiangular(5)
    isinstance(c, Coordinates)
    assert c.csize == 5**2
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with tuple
    c = samplings.equiangular((3, 5))
    assert c.csize == 3*5
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with spherical harmonic order
    c = samplings.equiangular(n_max=5)
    assert c.csize == 4 * (5 + 1)**2
    npt.assert_allclose(np.sum(c.weights), 1)

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.equiangular(5, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)


def test_gaussian():
    n_max = 1
    sampling = samplings.gaussian(n_max=n_max)
    assert isinstance(sampling, SamplingSphere)


def test_gaussian_pyfar():
    # test without parameters
    with raises(ValueError):
        samplings.gaussian()

    # test with single number of points
    c = samplings.gaussian(5)
    isinstance(c, Coordinates)
    assert c.csize == 5**2
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with tuple
    c = samplings.gaussian((3, 5))
    assert c.csize == 3*5
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with spherical harmonic order
    c = samplings.gaussian(n_max=5)
    assert c.csize == 2 * (5 + 1)**2
    npt.assert_allclose(np.sum(c.weights), 1)

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.gaussian(5, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)


def test_em32():
    sampling = samplings.eigenmike_em32()
    assert isinstance(sampling, SamplingSphere)


def test_icosahedron_ke4():
    sampling = samplings.icosahedron_ke4()
    assert isinstance(sampling, SamplingSphere)


def test_equalarea():
    sampling = samplings.equal_area(2)
    assert isinstance(sampling, SamplingSphere)


def test_spiral_points():
    sampling = samplings.spiral_points(2)
    assert isinstance(sampling, SamplingSphere)


def test_equal_angle():
    # test with tuple
    c = samplings.equal_angle((10, 20))
    assert isinstance(c, Coordinates)
    # test with number
    c = samplings.equal_angle(10)
    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)
    # test user radius
    c = samplings.equal_angle(10, 1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)

    # test assertions
    with raises(ValueError):
        c = samplings.equal_angle((11, 20))
    with raises(ValueError):
        c = samplings.equal_angle((20, 11))


def test_great_circle():
    # test with default values
    c = samplings.great_circle()
    assert isinstance(c, Coordinates)
    # check default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test if azimuth matching angles work
    c = samplings.great_circle(0, 4, match=90)
    azimuth = c.azimuth * 180 / np.pi
    for deg in [0, 90, 180, 270]:
        assert deg in azimuth

    # test user radius
    c = samplings.great_circle(radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)

    # test fractional azimuth resolution
    c = samplings.great_circle(60, 4,  azimuth_res=.1, match=90)
    npt.assert_allclose(c.azimuth[1] * 180 / np.pi, 7.5, atol=1e-15)

    # test assertion: 1 / azimuth_res is not an integer
    with raises(AssertionError):
        samplings.great_circle(azimuth_res=.6)
    # test assertion: 360 / match is not an integer
    with raises(AssertionError):
        samplings.great_circle(match=270)
    # test assertion: match / azimuth_res is not an integer
    with raises(AssertionError):
        samplings.great_circle(azimuth_res=.5, match=11.25)


def test_lebedev():
    # test without parameters
    assert samplings.lebedev() is None

    # test with degree
    c = samplings.lebedev(14)
    isinstance(c, Coordinates)
    assert c.csize == 14
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with spherical harmonic order
    c = samplings.lebedev(n_max=3)
    assert c.csize == 26
    npt.assert_allclose(np.sum(c.weights), 1)

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.lebedev(6, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)


def test_fliege():
    # test without parameters
    assert samplings.fliege() is None

    # test with degree
    c = samplings.fliege(16)
    isinstance(c, Coordinates)
    assert c.csize == 16
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with spherical harmonic order
    c = samplings.fliege(n_max=3)
    assert c.csize == 16
    npt.assert_allclose(np.sum(c.weights), 1)

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.fliege(4, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)

    # test exceptions
    with raises(ValueError):
        c = samplings.fliege(9, 2)
    with raises(ValueError):
        c = samplings.fliege(30)

