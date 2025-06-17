import numpy as np
import pytest

import spharpy.samplings as samplings
from spharpy import SamplingSphere
from pyfar import Coordinates
import numpy.testing as npt
from pytest import raises
from spharpy.spherical import (
    spherical_harmonic_basis_real, spherical_harmonic_basis)


def test_cube_equidistant_int():
    n_points = 3
    coords = samplings.cube_equidistant(n_points)
    x = np.tile(np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1]), 3)
    y = np.hstack((np.ones(9) * -1, np.zeros(9), np.ones(9)))
    z = np.tile(np.array([-1, 0, 1]), 9)
    np.testing.assert_allclose(x, coords.x)
    np.testing.assert_allclose(y, coords.y)
    np.testing.assert_allclose(z, coords.z)
    assert type(coords) is Coordinates
    assert coords.csize == 3**3


def test_cube_equidistant_tuple():
    # test with tuple
    c = samplings.cube_equidistant((3, 2, 4))
    assert c.csize == 3*2*4


def test_hyperinterpolation(download_sampling):
    n_max = 1
    download_sampling('hyperinterpolation', n_max)
    sampling = samplings.hyperinterpolation(n_max=n_max)
    assert sampling.radius.size == (n_max+1)**2


def test_hyperinterpolation_default_n_max():
    # check if n_max is set properly
    sampling = samplings.hyperinterpolation(n_points=4)
    assert sampling.n_max == 1
    assert isinstance(sampling.n_max, int)


def test_sph_extremal(download_sampling):
    # load test data
    download_sampling('hyperinterpolation', [1, 10])

    # test without parameters
    assert samplings.hyperinterpolation() is None

    # test with n_points
    c = samplings.hyperinterpolation(4)
    assert type(c) is SamplingSphere
    assert c.csize == 4

    # test with spherical harmonic order
    c = samplings.hyperinterpolation(n_max=1)
    assert c.csize == 4

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.hyperinterpolation(4, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)

    # test loading SH order > 9
    c = samplings.hyperinterpolation(n_max=10)

    # test quadrature
    npt.assert_allclose(np.sum(c.weights), 4 * np.pi)
    assert not c.quadrature

    # test exceptions
    with raises(ValueError):
        c = samplings.hyperinterpolation(4, 1)
    with raises(ValueError):
        c = samplings.hyperinterpolation(5)
    with raises(ValueError):
        c = samplings.hyperinterpolation(n_max=0)


def test_t_design_const_e(download_sampling):
    order = 2
    download_sampling('t-design', np.arange(1, 11))
    coords = samplings.t_design(
        n_max=order, criterion='const_energy')
    assert type(coords) is SamplingSphere


def test_t_design_const_angle(download_sampling):
    order = 2
    download_sampling('t-design', np.arange(1, 11))
    coords = samplings.t_design(
        n_max=order, criterion='const_angular_spread')
    assert type(coords) is SamplingSphere


def test_t_design_invalid(download_sampling):
    order = 2
    download_sampling('t-design', np.arange(1, 11))
    with pytest.raises(ValueError, match='Invalid design'):
        samplings.t_design(n_max=order, criterion='bla')


def test_sph_t_design(download_sampling):
    # load test data
    download_sampling('t-design', np.arange(1, 11))

    # test without parameters
    assert samplings.t_design() is None

    # test with degree
    c = samplings.t_design(2)
    isinstance(c, SamplingSphere)
    assert type(c) is SamplingSphere
    assert c.csize == 6

    # test with spherical harmonic order
    c = samplings.t_design(n_max=1)
    assert c.csize == 6
    c = samplings.t_design(
        n_max=1, criterion='const_angular_spread')
    assert c.csize == 8

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.t_design(2, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)

    # test loading degree order > 9
    c = samplings.t_design(10)

    # test quadrature
    assert not c.quadrature

    # test exceptions
    with raises(ValueError):
        c = samplings.t_design(4, 1)
    with raises(ValueError):
        c = samplings.t_design(0)
    with raises(ValueError):
        c = samplings.t_design(n_max=0)
    with raises(ValueError):
        c = samplings.t_design(2, criterion='const_thread')


def test_dodecahedron():
    sampling = samplings.dodecahedron()
    assert type(sampling) is SamplingSphere


def test_sph_dodecahedron():
    # test with default radius
    c = samplings.dodecahedron()
    assert type(c) is SamplingSphere
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test with user radius
    c = samplings.dodecahedron(1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)

    # test quadrature
    assert not c.quadrature


def test_icosahedron():
    sampling = samplings.icosahedron()
    assert type(sampling) is SamplingSphere

    # test quadrature
    assert not sampling.quadrature


def test_sph_icosahedron():
    # test with default radius
    c = samplings.icosahedron()
    assert type(c) is SamplingSphere
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test with user radius
    c = samplings.icosahedron(1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)


def test_equiangular():
    # test without parameters
    with raises(ValueError):
        samplings.equiangular()

    # test with single number of points
    c = samplings.equiangular(5)
    assert type(c) is SamplingSphere
    assert c.csize == 5**2

    # test with tuple
    c = samplings.equiangular((3, 5))
    assert c.csize == 3*5

    # test with spherical harmonic order
    c = samplings.equiangular(n_max=5)
    assert c.csize == 4 * (5 + 1)**2
    npt.assert_allclose(np.sum(c.weights), 4*np.pi)

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.equiangular(5, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)


@pytest.mark.parametrize("n_points", np.arange(2, 40, 2))
def test_equiangular_weights_n_points_even(n_points):
    sampling = samplings.equiangular(n_points=n_points)
    npt.assert_almost_equal(np.sum(sampling.weights), 4*np.pi)
    assert sampling.cshape == sampling.weights.shape
    assert sampling.cshape == n_points*n_points
    assert sampling.quadrature is True


@pytest.mark.parametrize("n_points", np.arange(1, 40, 2))
def test_equiangular_weights_n_points_odd(n_points):
    sampling = samplings.equiangular(n_points=n_points)
    assert sampling.weights is None
    assert sampling.cshape == n_points*n_points
    assert sampling.quadrature is False


@pytest.mark.parametrize(
    "n_points",
    [(5, 5), (4, 5), (4, 6)],
)
def test_equiangular_weights_n_points_tuple_invalid(n_points):
    sampling = samplings.equiangular(n_points=n_points)
    assert sampling.weights is None
    assert sampling.quadrature is False


def test_equiangular_weights_n_points_tuple_valid():
    n_points = (4, 4)
    sampling = samplings.equiangular(n_points=n_points)
    npt.assert_almost_equal(np.sum(sampling.weights), 4*np.pi)


@pytest.mark.parametrize("n_max", np.arange(1, 15))
def test_equiangular_weights_n_max(n_max):
    sampling = samplings.equiangular(n_max=n_max)
    npt.assert_almost_equal(np.sum(sampling.weights), 4*np.pi)
    assert sampling.cshape == sampling.weights.shape
    assert sampling.cshape == 4*(n_max+1)**2
    assert type(sampling) is SamplingSphere


@pytest.mark.parametrize(
    'basis_func', [
        spherical_harmonic_basis, spherical_harmonic_basis_real
    ])
def test_equiangular_orthogonality(basis_func):
    n_max = 4
    sampling = samplings.equiangular(n_max=n_max)

    Y = basis_func(n_max, sampling)
    npt.assert_allclose(
        Y.conj().T @ np.diag(sampling.weights) @ Y,
        np.eye((n_max+1)**2),
        atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("n_points", np.arange(1, 40))
def test_gaussian_weights_n_points(n_points):
    sampling = samplings.gaussian(n_points=n_points)
    npt.assert_almost_equal(np.sum(sampling.weights), 4*np.pi)
    assert sampling.cshape == sampling.weights.shape
    assert sampling.cshape == 2*n_points*n_points
    assert type(sampling) is SamplingSphere


@pytest.mark.parametrize("n_max", np.arange(1, 15))
def test_gaussian_weights_n_max(n_max):
    sampling = samplings.gaussian(n_max=n_max)
    npt.assert_almost_equal(np.sum(sampling.weights), 4*np.pi)
    assert sampling.cshape == sampling.weights.shape
    assert sampling.cshape == 2*(n_max+1)*(n_max+1)
    assert type(sampling) is SamplingSphere


@pytest.mark.parametrize(
    'basis_func', [
        spherical_harmonic_basis, spherical_harmonic_basis_real
    ])
def test_gaussian_orthogonality(basis_func):
    n_max = 4
    sampling = samplings.gaussian(n_max=n_max)

    Y = basis_func(n_max, sampling)
    npt.assert_allclose(
        Y.conj().T @ np.diag(sampling.weights) @ Y,
        np.eye((n_max+1)**2),
        atol=1e-6, rtol=1e-6
    )


def test_gaussian():
    # test without parameters
    with raises(ValueError):
        samplings.gaussian()

    # n_points must be a positive natural number
    with raises(ValueError, match='positive natural number'):
        samplings.gaussian(n_points=(2, 2))

    # n_points must be a positive natural number
    with raises(ValueError, match='positive natural number'):
        samplings.gaussian(n_points=3.2)

    # test with single number of points
    c = samplings.gaussian(5)
    assert type(c) is SamplingSphere
    assert c.csize == 5*(5*2)
    npt.assert_allclose(np.sum(c.weights), 4*np.pi)

    # test with spherical harmonic order
    c = samplings.gaussian(n_max=5)
    assert c.csize == 2 * (5 + 1)**2
    npt.assert_allclose(np.sum(c.weights), 4*np.pi)

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.gaussian(5, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)

    # test quadrature
    npt.assert_allclose(np.sum(c.weights), 4 * np.pi)
    assert c.quadrature


def test_em32():
    sampling = samplings.eigenmike_em32()
    assert type(sampling) is SamplingSphere


def test_icosahedron_ke4():
    sampling = samplings.icosahedron_ke4()
    assert type(sampling) is SamplingSphere

    # test quadrature
    assert not sampling.quadrature


def test_equalarea():
    sampling = samplings.equal_area(2)
    assert type(sampling) is SamplingSphere

    # test quadrature
    assert not sampling.quadrature


def test_spiral_points():
    sampling = samplings.spiral_points(2)
    assert type(sampling) is SamplingSphere

    # test quadrature
    assert not sampling.quadrature


def test_equal_angle():
    # test with tuple
    c = samplings.equal_angle((10, 20))
    assert type(c) is SamplingSphere
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

    # test quadrature
    assert not c.quadrature


def test_great_circle():
    # test with default values
    c = samplings.great_circle()
    assert type(c) is SamplingSphere
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

    # test quadrature
    assert not c.quadrature

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
    assert type(c) is SamplingSphere
    assert c.csize == 14

    # test with spherical harmonic order
    c = samplings.lebedev(n_max=3)
    assert c.csize == 26

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.lebedev(6, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)

    # test quadrature
    npt.assert_allclose(np.sum(c.weights), 4 * np.pi)
    assert c.quadrature


def test_fliege():
    # test without parameters
    assert samplings.fliege() is None

    # test with degree
    c = samplings.fliege(16)
    assert type(c) is SamplingSphere
    assert c.csize == 16

    # test with spherical harmonic order
    c = samplings.fliege(n_max=3)
    assert c.csize == 16

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.fliege(4, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)

    # test quadrature
    npt.assert_allclose(np.sum(c.weights), 4 * np.pi)
    assert not c.quadrature

    # test exceptions
    with raises(ValueError):
        c = samplings.fliege(9, 2)
    with raises(ValueError):
        c = samplings.fliege(30)


def test_em64():
    sampling = samplings.eigenmike_em64()
    assert type(sampling) is SamplingSphere

    npt.assert_allclose(
        np.sum(sampling.weights), 4*np.pi, atol=1e-6, rtol=1e-6)
