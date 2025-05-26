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


def test_error_multiple_radius_initialization():
    """
    Test if entering points with multiple radii during initialization raises
    an error.
    """

    match = '0.5 m, which exceeds the tolerance of 1e-06 m'
    with pytest.raises(ValueError, match=match):
        SamplingSphere([1, 0], 0, 0)


def test_error_multiple_radius_setter():
    """
    Test if entering points with multiple radii after initialization raises
    an error.
    """

    sampling_sphere = SamplingSphere([1, 1], 0, 0)

    match = '0.5 m, which exceeds the tolerance of 1e-06 m'
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


@pytest.mark.parametrize(['tolerance'], [
    (None, ), ([0, 1], ), (np.array([0, 1]), ), (-.1, )
])
def test_radius_tolerance_input(tolerance):
    """Test if passing wrong values raises the expected error"""

    match = 'The radius tolerance must be a number greater than zero'
    with pytest.raises(ValueError, match=match):
        SamplingSphere([1, 1], 0, 0, radius_tolerance=tolerance)
