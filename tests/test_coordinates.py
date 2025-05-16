from spharpy import SamplingSphere
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

    match = '0.5 m, which exceeds the tolerance of 5e-15 m'
    with pytest.raises(ValueError, match=match):
        SamplingSphere([1, 0], 0, 0)


def test_error_multiple_radius_setter():
    """
    Test if entering points with multiple radii after initialization raises
    an error.
    """

    sampling_sphere = SamplingSphere([1, 1], 0, 0)

    match = '0.5 m, which exceeds the tolerance of 5e-15 m'
    with pytest.raises(ValueError, match=match):
        sampling_sphere.x = [1, 0]
