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

    # weights do not sum to 4 pi
    sampling.weights = [1, 1]
    with pytest.raises(ValueError, match="quadrature can not be True"):
        sampling.quadrature = True

    # negative weight
    sampling.weights = [-2 * np.pi, 6 * np.pi]
    with pytest.raises(ValueError, match="quadrature can not be True"):
        sampling.quadrature = True
