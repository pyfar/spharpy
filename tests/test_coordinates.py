from spharpy import SamplingSphere
import numpy as np
import pytest


def test_sampling_sphere_init():
    sampling = SamplingSphere()
    assert isinstance(sampling, SamplingSphere)


def test_sampling_sphere_init_value():
    sampling = SamplingSphere(1, 0, 0, 0)
    assert isinstance(sampling, SamplingSphere)

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


@pytest.mark.parametrize(['sampling'], [
    (SamplingSphere([1, 1], 0, 0), ),
    (SamplingSphere.from_cartesian([1, 1], 0, 0), ),
    (SamplingSphere.from_spherical_elevation([0, 0], 0, 1), ),
    (SamplingSphere.from_spherical_colatitude([0, 0], 0, 1), ),
    (SamplingSphere.from_spherical_side([0, 0], 0, 1), ),
    (SamplingSphere.from_spherical_front([0, 0], 0, 1), ),
    (SamplingSphere.from_cylindrical([0, 0], 0, 1), ),
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


@pytest.mark.parametrize(['tolerance'], [
    (None, ), ([0, 1], ), (np.array([0, 1]), ), (-.1, )
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
    with pytest.raises(ValueError, match=message):
        weights_invalid = np.ones(6)*4*np.pi/6
        weights_invalid[0] = np.nan
        sampling.weights = weights_invalid

    with pytest.raises(ValueError, match=message):
        weights_invalid = np.ones(6)*4*np.pi/6 * -1
        weights_invalid[0] = -1
        sampling.weights = weights_invalid


def test_quadrature_default_setter_getter():
    """Test the default value, setter, and getter for quadrature."""

    weights = [2 * np.pi, 2 * np.pi]
    sampling = SamplingSphere([1, 1], 0, 0, weights=weights)

    # test default value and getter
    assert sampling.quadrature == False

    # test setter and getter
    sampling.quadrature = True
    assert sampling.quadrature == True


def test_quadrature_setter_errors():
    """Test errors in the quadrature setter for wrong input data."""

    sampling = SamplingSphere([1, 1], 0, 0)

    # input type
    with pytest.raises(TypeError, match="True or False but is None"):
        sampling.quadrature = None

    # weights are None
    with pytest.raises(ValueError, match="quadrature can not be True"):
        sampling.quadrature = True
