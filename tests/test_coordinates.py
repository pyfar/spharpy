from spharpy import SamplingSphere
from pyfar import Coordinates
import numpy as np
import numpy.testing as npt
import pytest
from spharpy.samplings import gaussian


def test_sampling_sphere_init():
    sampling = SamplingSphere()
    assert isinstance(sampling, SamplingSphere)


def test_sampling_sphere_init_value():
    sampling = SamplingSphere(1, 0, 0, 0)
    assert isinstance(sampling, SamplingSphere)


def test_sampling_sphere_from_coordinates():
    """Test converting Coordinates to SamplingSphere."""

    coordinates = Coordinates([1, 0, -1, 0], [0, 1, 0, -1], 0,
                              weights=[np.pi, np.pi, np.pi, np.pi])
    sampling_sphere = SamplingSphere.from_coordinates(coordinates)

    # check data in sampling_sphere
    assert type(sampling_sphere) is SamplingSphere
    npt.assert_equal(sampling_sphere.cartesian, coordinates.cartesian)
    npt.assert_equal(sampling_sphere.weights, coordinates.weights)
    assert sampling_sphere.n_max is None
    assert sampling_sphere.radius_tolerance == 1e-6
    assert sampling_sphere.quadrature_tolerance == 1e-10
    assert sampling_sphere.comment == coordinates.comment

    # make sure mutable arrays are copied from coordinates
    coordinates.weights = None
    assert sampling_sphere.weights is not None

    coordinates.x = [1, 1, 1, 1]
    npt.assert_equal(sampling_sphere.x, [1, 0, -1, 0])

    coordinates.y = [1, 1, 1, 1]
    npt.assert_equal(sampling_sphere.y, [0, 1, 0, -1])

    coordinates.z = [1, 1, 1, 1]
    npt.assert_equal(sampling_sphere.z, [0, 0, 0, 0])


def sampling_cube():
    """Helper function returning a cube sampling."""
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


def test_error_multiple_radius_initialization():
    """
    Test if entering points with multiple radii during initialization raises
    an error.
    """

    match = '1 m, which exceeds the tolerance of 1e-06 m'
    with pytest.raises(ValueError, match=match):
        SamplingSphere([1, 0], 0, 0)


def test_error_multiple_radius_setter():
    """
    Test if entering points with multiple radii after initialization raises
    an error.
    """

    sampling_sphere = SamplingSphere([1, 1], 0, 0)

    match = '1 m, which exceeds the tolerance of 1e-06 m'
    with pytest.raises(ValueError, match=match):
        sampling_sphere.x = [1, 0]


@pytest.mark.parametrize('sampling', [
    SamplingSphere([1, 1], 0, 0),
    SamplingSphere.from_cartesian([1, 1], 0, 0),
    SamplingSphere.from_spherical_elevation([0, 0], 0, 1),
    SamplingSphere.from_spherical_colatitude([0, 0], 0, 1),
    SamplingSphere.from_spherical_side([0, 0], 0, 1),
    SamplingSphere.from_spherical_front([0, 0], 0, 1),
    SamplingSphere.from_cylindrical([0, 0], 0, 1),
])
def test_radius_tolerance(sampling):
    """
    Test getter and setter for radius tolerance and the related error message.
    """
    tolerance = 1e-3

    # test default value
    assert sampling.radius_tolerance == 1e-6
    # change tolerance
    sampling.radius_tolerance = tolerance
    assert sampling.radius_tolerance == tolerance

    with pytest.raises(ValueError, match=f'{tolerance:.3g}'):
        sampling.x = [0, 1]


def test_radius_tolerance_error():
    """
    Test if setting the radius tolerance too strict raises an error for
    existing data.
    """
    sampling = SamplingSphere([1, 1.1], 0, 0, radius_tolerance=.2)

    with pytest.raises(ValueError, match='the tolerance of 0.01'):
        sampling.radius_tolerance = .01


@pytest.mark.parametrize('tolerance', [
    None, [0, 1], np.array([0, 1]), -.1,
])
def test_radius_tolerance_input(tolerance):
    """Test if passing wrong values raises the expected error."""

    match = 'The radius tolerance must be a number greater than zero'
    with pytest.raises(ValueError, match=match):
        SamplingSphere([1, 1], 0, 0, radius_tolerance=tolerance)


def test_weights_getter():
    x, y, z = sampling_cube()
    n_max = 1
    weights = np.ones(6)*4*np.pi/6
    sampling = SamplingSphere(x, y, z, n_max, weights=weights)

    np.testing.assert_allclose(sampling.weights, weights)


def test_setting_weights():
    x, y, z = sampling_cube()
    n_max = 1
    sampling = SamplingSphere(x, y, z, n_max)

    weights = np.ones(6)*4*np.pi/6
    sampling.weights = weights

    np.testing.assert_array_equal(sampling._weights, weights)
    assert sampling._weights.shape == (6,)


def test_setting_weights_invalid():
    x, y, z = sampling_cube()
    n_max = 1
    sampling = SamplingSphere(x, y, z, n_max)

    message = r"The sum of the weights must be equal to 4\*pi."
    with pytest.raises(ValueError, match=message):
        sampling.weights = np.ones(6)/6

    with pytest.raises(ValueError, match=message):
        sampling._set_weights(np.ones(6)/6)

    message = "All weights must be positive."
    weights_invalid = np.ones(6)*4*np.pi/6
    weights_invalid[0] = np.nan
    with pytest.raises(ValueError, match=message):
        sampling.weights = weights_invalid

    weights_invalid = np.ones(6)*4*np.pi/6 * -1
    weights_invalid[0] = -1
    with pytest.raises(ValueError, match=message):
        sampling.weights = weights_invalid


def test_quadrature_getter_changing_weights():
    # create a quadrature grid (gaussian)
    sampling = gaussian(n_max=3)
    # check if quadrature is set properly
    assert sampling.quadrature
    # update weights such that quadrature requirement is not valid anymore
    weights = 4 * np.pi * np.ones(sampling.cshape) / sampling.cshape
    sampling.weights = weights
    assert not sampling.quadrature


def test_quadrature_getter_changing_points():
    # create a quadrature grid (gaussian)
    sampling = gaussian(n_max=3)
    # check if quadrature is set properly
    assert sampling.quadrature
    # update points such that quadrature is not valid anymore
    rng = np.random.default_rng()
    sampling.spherical_colatitude = np.concatenate([
        rng.random((sampling.csize, 1)),
        np.ones((sampling.csize, 1)),
        np.ones((sampling.csize, 1))], axis=1)
    assert not sampling.quadrature


def test_repr():
    """Test representation string."""

    sampling = SamplingSphere([1, -1], 0, 0)
    repr_str = sampling.__repr__()
    assert repr_str == 'SamplingSphere: n_max=None, cshape=(2,)'

    sampling = SamplingSphere([1, -1], 0, 0, n_max=0)
    repr_str = sampling.__repr__()
    assert repr_str == 'SamplingSphere: n_max=0, cshape=(2,)'


def test_repr_empty():
    sampling = SamplingSphere()
    assert sampling.__repr__() == 'SamplingSphere: empty'


def test_empty_object():
    sampling = SamplingSphere()
    assert sampling.cshape == (0,)
    with pytest.raises(ValueError, match='Object is empty.'):
        _ = sampling.cartesian
