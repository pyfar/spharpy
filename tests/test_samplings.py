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
