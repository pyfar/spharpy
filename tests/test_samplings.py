import numpy as np
import pytest

import spharpy.samplings as samplings
from spharpy import SamplingSphere
from pyfar import Coordinates
import numpy.testing as npt
from spharpy.spherical import (
    spherical_harmonic_basis_real, spherical_harmonic_basis)


@pytest.mark.parametrize("flatten_output", [True, False])
def test_equidistant_cuboid_sampling_int(flatten_output):
    n_points = 3
    coords = samplings.equidistant_cuboid(
        n_points, flatten_output=flatten_output)
    data = np.linspace(-1, 1, 3)
    x, y, z = np.meshgrid(data, data, data, indexing='ij')
    if flatten_output:
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
    np.testing.assert_allclose(x, coords.x)
    np.testing.assert_allclose(y, coords.y)
    np.testing.assert_allclose(z, coords.z)
    assert type(coords) is Coordinates
    assert coords.csize == 3**3
    if flatten_output:
        assert coords.cshape == (3*3*3,)
    else:
        assert coords.cshape == (3, 3, 3)


def test_equidistant_cuboid_sampling_tuple():
    c = samplings.equidistant_cuboid((2, 3, 4), flatten_output=False)
    assert c.csize == 2*3*4
    assert c.cshape == (2, 3, 4)
    npt.assert_allclose(c.x[0], c.x[0, 0, 0])
    npt.assert_allclose(c.y[:, 0], c.y[0, 0, 0])
    npt.assert_allclose(c.z[:, :, 0], c.z[0, 0, 0])


def test_equidistant_cuboid_sampling_tuple_flatten():
    c = samplings.equidistant_cuboid((2, 3, 4), flatten_output=True)
    assert c.csize == 2*3*4
    assert c.cshape == (2*3*4,)


def test_equidistant_cuboid_sampling_invalid():
    with pytest.raises(ValueError, match='flatten_output must be a boolean.'):
        samplings.equidistant_cuboid(3, flatten_output='bla')
    with pytest.raises(ValueError, match='The number of points needs to be'):
        samplings.equidistant_cuboid(-3)
    with pytest.raises(ValueError, match='The number of points needs to be'):
        samplings.equidistant_cuboid((3, -3, 3))
    with pytest.raises(ValueError, match='The number of points needs to be'):
        samplings.equidistant_cuboid((3, 3, 3.2))
    with pytest.raises(ValueError, match='The number of points needs to be'):
        samplings.equidistant_cuboid((3, 3, -3))


def test_t_design_const_e(download_sampling):
    n_max = 2
    degree = 2 * n_max
    download_sampling('t-design', np.arange(1, 11))
    coords = samplings.t_design(
        n_max, criterion='const_energy')
    assert type(coords) is SamplingSphere
    assert coords.n_max == 2
    assert coords.csize == int(np.ceil((degree + 1)**2 / 2) + 1)


def test_t_design_const_angle(download_sampling):
    n_max = 2
    download_sampling('t-design', np.arange(1, 11))
    coords = samplings.t_design(
        n_max, criterion='const_angular_spread')
    assert type(coords) is SamplingSphere
    assert coords.csize == 18


def test_t_design_invalid(download_sampling):
    download_sampling('t-design', np.arange(1, 11))
    with pytest.raises(ValueError, match='Invalid design'):
        samplings.t_design(2, criterion='bla')


def test_t_design_limits_const_energy(download_sampling):
    download_sampling('t-design', [179, 180])
    samplings.t_design(90, criterion='const_energy')
    with pytest.raises(
            ValueError,
            match='n_max must be between 1 and 90 for const_energy'):
        samplings.t_design(91, criterion='const_energy')
    with pytest.raises(
            ValueError,
            match='n_max must be between 1 and 90 for const_energy'):
        samplings.t_design(0, criterion='const_energy')


def test_t_design_limits_const_angular_spread(download_sampling):
    download_sampling('t-design', [179, 180])
    samplings.t_design(89, criterion='const_angular_spread')
    with pytest.raises(
            ValueError,
            match='n_max must be between 1 and 89 for const_angular_spread'):
        samplings.t_design(90, criterion='const_angular_spread')
    with pytest.raises(
            ValueError,
            match='n_max must be between 1 and 89 for const_angular_spread'):
        samplings.t_design(0, criterion='const_angular_spread')


def test_t_design_n_max_error(download_sampling):
    download_sampling('t-design', np.arange(1, 11))
    samplings.t_design(89)
    with pytest.raises(
            ValueError,
            match='n_max must be an integer'):
        samplings.t_design(1.5)


def test_sph_t_design(download_sampling):
    # load test data
    download_sampling('t-design', np.arange(1, 11))

    # test with n_max
    c = samplings.t_design(1)
    isinstance(c, SamplingSphere)
    assert type(c) is SamplingSphere
    assert c.csize == 6

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
    with pytest.raises(ValueError, match='Invalid design criterion'):
        samplings.t_design(2, criterion='const_thread')
    with pytest.raises(ValueError, match='radius must be a positive number'):
        samplings.t_design(2, radius=-1)
    with pytest.raises(ValueError, match='radius must be a positive number'):
        samplings.t_design(2, radius='test')


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
    with pytest.raises(ValueError, match='Either the n_points or n_max needs'):
        samplings.equiangular()

    # test with single number of points
    c = samplings.equiangular(n_points=5)
    assert type(c) is SamplingSphere
    assert c.csize == 5**2

    # test with tuple
    c = samplings.equiangular(n_points=(3, 5))
    assert c.csize == 3*5

    # test with spherical harmonic order
    c = samplings.equiangular(5)
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
    assert sampling.quadrature


@pytest.mark.parametrize("n_points", np.arange(1, 40, 2))
def test_equiangular_weights_n_points_odd(n_points):
    sampling = samplings.equiangular(n_points=n_points)
    assert sampling.weights is None
    assert sampling.cshape == n_points*n_points
    assert not sampling.quadrature


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
    'basis_func', [spherical_harmonic_basis, spherical_harmonic_basis_real])
def test_equiangular_orthogonality(basis_func):
    n_max = 4
    sampling = samplings.equiangular(n_max=n_max)

    Y = basis_func(n_max, sampling)
    npt.assert_allclose(
        Y.conj().T @ np.diag(sampling.weights) @ Y,
        np.eye((n_max+1)**2),
        atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("n_max", np.arange(1, 15))
def test_gaussian_weights_n_max(n_max):
    sampling = samplings.gaussian(n_max=n_max)
    npt.assert_almost_equal(np.sum(sampling.weights), 4*np.pi)
    assert sampling.cshape == sampling.weights.shape
    assert sampling.cshape == 2*(n_max+1)*(n_max+1)
    assert type(sampling) is SamplingSphere


@pytest.mark.parametrize(
    'basis_func', [spherical_harmonic_basis, spherical_harmonic_basis_real])
def test_gaussian_orthogonality(basis_func):
    n_max = 4
    sampling = samplings.gaussian(n_max=n_max)

    Y = basis_func(n_max, sampling)
    npt.assert_allclose(
        Y.conj().T @ np.diag(sampling.weights) @ Y,
        np.eye((n_max+1)**2),
        atol=1e-6, rtol=1e-6)


def test_gaussian_quadrature():
    n_max = 3
    sampling = samplings.gaussian(n_max=n_max)

    assert sampling.quadrature


def test_gaussian():

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
    with pytest.raises(ValueError,
                       match='delta_phi must be an integer divisor'):
        c = samplings.equal_angle((11, 20))
    with pytest.raises(ValueError,
                       match='delta_theta must be an integer divisor'):
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
    with pytest.raises(AssertionError):
        samplings.great_circle(azimuth_res=.6)
    # test assertion: 360 / match is not an integer
    with pytest.raises(AssertionError):
        samplings.great_circle(match=270)
    # test assertion: match / azimuth_res is not an integer
    with pytest.raises(AssertionError):
        samplings.great_circle(azimuth_res=.5, match=11.25)


def test_lebedev(capfd):
    # test without parameters
    assert samplings.lebedev() is None
    # test command line output
    out, _ = capfd.readouterr()
    assert 'Possible input values' in out
    assert 'SH order 1, number of points 6' in out

    # test with spherical harmonic order
    c = samplings.lebedev(3)
    assert c.csize == 26
    assert type(c) is SamplingSphere

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)

    # test user radius
    c = samplings.lebedev(1, radius=1.5)
    npt.assert_allclose(c.radius, 1.5, atol=1e-15)


@pytest.mark.parametrize("degree", np.array([
    6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302], dtype=int))
def test_lebedev_orthogonality(degree):
    """
    Test orthogonality of the transform.

    This was done after discovering https://github.com/pyfar/spharpy/issues/276
    to make sure that the sampling can be used for SH transforms using the
    pseudo inverse.

    The test for all degrees takes too long and was only done once. If required
    replace the above with

    @pytest.mark.parametrize("degree", np.array([
    6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434,
    590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334,
    4802, 5294, 5810], dtype=int))
    """

    n_max = int(np.sqrt(degree / 1.3) - 1)
    sampling = samplings.lebedev(n_max)
    sampling.weights = samplings.calculate_sampling_weights(sampling)
    Y = spherical_harmonic_basis_real(n_max, sampling)

    npt.assert_allclose(
        Y.T @ np.diag(sampling.weights) @ Y,
        np.eye((n_max + 1)**2),
        atol=1e-2, rtol=1e-2)


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
    with pytest.raises(ValueError, match='n_points or n_max must be None'):
        c = samplings.fliege(9, 2)
    with pytest.raises(ValueError, match='Invalid number of points n_points'):
        c = samplings.fliege(30)


def test_em64():
    sampling = samplings.eigenmike_em64()
    assert type(sampling) is SamplingSphere

    npt.assert_allclose(
        np.sum(sampling.weights), 4*np.pi, atol=1e-6, rtol=1e-6)


def test_hyperinterpolation_default(download_sampling):
    n_max = 1
    # download just required sampling for testing
    download_sampling('hyperinterpolation', [n_max])

    c = samplings.hyperinterpolation(1)

    # test sampling properties
    assert type(c) is SamplingSphere
    assert c.n_max == n_max
    assert c.csize == (n_max+1)**2
    assert c.radius.size == (n_max+1)**2

    # test default radius
    npt.assert_allclose(c.radius, 1, atol=1e-15)
    npt.assert_allclose(np.sum(c.weights), 4 * np.pi)


@pytest.mark.parametrize("radius", [1, 5])
def test_hyperinterpolation_radius(download_sampling, radius):
    download_sampling('hyperinterpolation', [1])
    sampling = samplings.hyperinterpolation(1, radius=radius)
    assert type(sampling) is SamplingSphere
    npt.assert_allclose(sampling.radius, radius, atol=1e-15)


@pytest.mark.parametrize("n_max", [-1, 'one'])
def test_hyperinterpolation_errors_n_max(n_max):
    with pytest.raises(
            ValueError, match='n_max must be an integer between 1 and 200'):
        samplings.hyperinterpolation(n_max)


@pytest.mark.parametrize("radius", [-1, 'one'])
def test_hyperinterpolation_errors_radius(radius):
    with pytest.raises(
            ValueError, match='radius must be a single positive value'):
        samplings.hyperinterpolation(1, radius)


def test_sph_gaussian_higher_order():
    s = samplings.gaussian(n_max=121)
    assert s.csize == (2 * (121 + 1)**2)
    npt.assert_allclose(np.sum(s.weights), 4*np.pi)
