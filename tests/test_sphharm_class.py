"""
Tests for spherical harmonic class
"""
import pytest
import numpy as np
import pyfar as pf
from spharpy.spherical import SphericalHarmonics
from spharpy.samplings import gaussian

def test_sphharm_init():
    coordinates = pf.Coordinates(1, 0, 0)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    assert sph_harm.n_max == 2
    assert np.all(sph_harm.coordinates == coordinates)

def test_sphharm_init_invalid_coordinates():
    with pytest.raises(TypeError):
        SphericalHarmonics(n_max=2, coordinates=[0, 0, 1])

def test_sphharm_init_invalid_n_max():
    coordinates = pf.Coordinates(1, 0, 0)
    with pytest.raises(ValueError):
        SphericalHarmonics(n_max=-1, coordinates=coordinates)

def test_sphharm_compute_basis():
    coordinates = gaussian(n_points=8)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    assert sph_harm.basis is not None

def test_sphharm_compute_basis_gradient():
    coordinates = pf.Coordinates(1, 0, 0)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    assert sph_harm.basis_gradient_theta is not None
    assert sph_harm.basis_gradient_phi is not None
def test_sphharm_compute_inverse_quad():
    coordinates = gaussian(n_points=4)
    # coordinates = pf.Coordinates(1, 0, 0)
    coordinates = pf.Coordinates.from_cartesian(coordinates.x, coordinates.y, coordinates.z)
    sh = SphericalHarmonics(2, coordinates, inverse_transform = 'quadrature')
    assert sh.basis_inv is not None

def test_sphharm_compute_inverse_pseudo_inv():
    coordinates = gaussian(n_points= 5)
    # coordinates = pf.Coordinates(1, 0, 0)
    coordinates = pf.Coordinates.from_cartesian(coordinates.x, coordinates.y, coordinates.z)
    sh = SphericalHarmonics(2, coordinates, inverse_transform = 'pseudo_inverse')
    assert sh.basis_inv is not None

def test_compute_basis_caching():
    n_max = 2
    points = np.random.rand(10, 3)
    coordinates = pf.Coordinates(points[:, 0], points[:, 1], points[:, 2], 'cart')
    sh = SphericalHarmonics(n_max, coordinates)

    # Call the method once and store the result
    initial_result = sh.basis

    # Change a property that affects the output of _compute_basis()
    sh.n_max = 3

    new_result = sh.basis

    # Call the method again and check that the result is different (cache miss)
    assert new_result is not initial_result

