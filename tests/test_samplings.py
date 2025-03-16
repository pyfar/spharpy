""" Tests for spatial sampling functions """

import numpy as np
import pytest

import spharpy.samplings as samplings
from spharpy.samplings.coordinates import Coordinates, SamplingSphere


def test_cube_equidistant():
    n_points = 3
    coords = samplings.cube_equidistant(n_points)
    x = np.tile(np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1]), 3)
    y = np.hstack((np.ones(9) * -1, np.zeros(9), np.ones(9)))
    z = np.tile(np.array([-1, 0, 1]), 9)
    np.testing.assert_allclose(x, coords.x)
    np.testing.assert_allclose(y, coords.y)
    np.testing.assert_allclose(z, coords.z)


def test_hyperinterpolation():
    n_max = 1
    sampling = samplings.hyperinterpolation(n_max)
    assert sampling.radius.size == (n_max+1)**2


def test_spherical_t_design_const_e():
    order = 2
    coords = samplings.spherical_t_design(
        order, criterion='const_energy')
    assert isinstance(coords, SamplingSphere)


def test_spherical_t_design_const_angle():
    order = 2
    coords = samplings.spherical_t_design(
        order, criterion='const_angular_spread')
    assert isinstance(coords, SamplingSphere)


def test_spherical_t_design_invalid():
    order = 2
    with pytest.raises(ValueError, match='Invalid design'):
        samplings.spherical_t_design(order, criterion='bla')


def test_dodecahedron():
    sampling = samplings.dodecahedron()
    assert isinstance(sampling, SamplingSphere)


def test_icosahedron():
    sampling = samplings.icosahedron()
    assert isinstance(sampling, SamplingSphere)


def test_equiangular():
    n_max = 1
    sampling = samplings.equiangular(n_max)
    assert isinstance(sampling, SamplingSphere)


def test_gaussian():
    n_max = 1
    sampling = samplings.gaussian(n_max)
    assert isinstance(sampling, SamplingSphere)


def test_em32():
    sampling = samplings.eigenmike_em32()
    assert isinstance(sampling, SamplingSphere)


def test_icosahedron_ke4():
    sampling = samplings.icosahedron_ke4()
    assert isinstance(sampling, SamplingSphere)


def test_equalarea():
    sampling = samplings.equalarea(2)
    assert isinstance(sampling, SamplingSphere)


def test_spiral_points():
    sampling = samplings.spiral_points(2)
    assert isinstance(sampling, SamplingSphere)


def test_em64():
    sampling = samplings.eigenmike_em64()
    assert isinstance(sampling, SamplingSphere)

    # weights = np.array([
    #     0.954, 0.9738, 1.0029, 1.0426, 1.0426, 1.0024, 0.9738, 0.954, 1.009,
    #     0.9932, 1.0024, 1.0324, 0.954, 1.0024, 1.0079, 1.0268, 1.0151, 0.9463,
    #     1.012, 1.0253, 1.009, 0.9932, 1.0324, 1.0151, 0.954, 1.0079, 1.0029,
    #     1.0024, 1.0268, 0.9463, 1.012, 1.0253, 0.954, 0.9738, 1.0029, 1.0426,
    #     1.0426, 1.0024, 0.954, 0.9738, 1.0268, 1.0151, 1.012, 0.9463, 1.0253,
    #     1.009, 0.9932, 1.0024, 1.0324, 1.0029, 0.954, 1.0024, 1.0324, 1.0151,
    #     0.954, 1.0079, 1.0024, 1.0079, 1.0268, 1.012, 0.9463, 1.009, 1.0253,
    #     0.9932,
    # ]) / 64 * 4 * np.pi

    # # check the individual weights
    # np.testing.assert_allclose(sampling.weights, weights)

    # # check if the weigths sum up to 4*pi to ensure valid integration on the unit sphere
    # np.testing.assert_allclose(np.sum(sampling.weights), 4*np.pi, atol=1e-4, rtol=1e-4)
