"""
Tests for spherical harmonic class
"""
import pytest
import numpy as np
import pyfar as pf
from spharpy.spherical import SphericalHarmonics
from spharpy.samplings import gaussian
def test_sphharm_init():
    coords = pf.Coordinates(1, 0, 0)
    sph_harm = SphericalHarmonics(n_max=2, coords=coords)
    assert sph_harm.n_max == 2
    assert np.all(sph_harm.coords == coords)

def test_sphharm_init_invalid_coords():
    with pytest.raises(TypeError):
        SphericalHarmonics(n_max=2, coords=[0, 0, 1])

def test_sphharm_init_invalid_n_max():
    coords = pf.Coordinates(1, 0, 0)
    with pytest.raises(ValueError):
        SphericalHarmonics(n_max=-1, coords=coords)

def test_sphharm_compute_basis():
    coords = pf.Coordinates(1, 0, 0)
    sph_harm = SphericalHarmonics(n_max=2, coords=coords)
    sph_harm.compute_basis()
    assert sph_harm.basis is not None

def test_sphharm_compute_basis_gradient():
    coords = pf.Coordinates(1, 0, 0)
    sph_harm = SphericalHarmonics(n_max=2, coords=coords)
    sph_harm.compute_basis_gradient()
    assert sph_harm.basis_gradient_theta is not None
    assert sph_harm.basis_gradient_phi is not None

def test_sphharm_compute_inverse_quad():
    coords = gaussian(n_max=3)
    # coords = pf.Coordinates(1, 0, 0)
    coords = pf.Coordinates.from_cartesian(coords.x, coords.y, coords.z)
    sh = SphericalHarmonics(2, coords, inverse_transform = 'quadrature')
    sh.compute_inverse()
    assert sh.basis_inv is not None

def test_sphharm_compute_inverse_pseudo_inv():
    coords = gaussian(n_max=3)
    # coords = pf.Coordinates(1, 0, 0)
    coords = pf.Coordinates.from_cartesian(coords.x, coords.y, coords.z)
    sh = SphericalHarmonics(2, coords, inverse_transform = 'pseudo_inverse')
    sh.compute_inverse()
    assert sh.basis_inv is not None

def test_sphharm_compute_inverse_gradient_quad():
    coords = gaussian(n_max=3)
    # coords = pf.Coordinates(1, 0, 0)
    coords = pf.Coordinates.from_cartesian(coords.x, coords.y, coords.z)
    sh = SphericalHarmonics(2, coords, inverse_transform = 'quadrature')
    sh.compute_inverse_gradient()
    assert sh.basis_inv_gradient_theta is not None
    assert sh.basis_inv_gradient_phi is not None

def test_sphharm_compute_inverse_gradient_pseudo_inv():
    coords = gaussian(n_max=3)
    # coords = pf.Coordinates(1, 0, 0)
    coords = pf.Coordinates.from_cartesian(coords.x, coords.y, coords.z)
    sh = SphericalHarmonics(2, coords, inverse_transform= 'pseudo_inverse')
    sh.compute_inverse_gradient()
    assert sh.basis_inv_gradient_theta is not None
    assert sh.basis_inv_gradient_phi is not None

def test_compute_basis_caching():
    n_max = 2
    points = np.random.rand(10, 3)
    coords = pf.Coordinates(points[:, 0], points[:, 1], points[:, 2], 'cart')
    sh = SphericalHarmonics(n_max, coords)

    # Call the method once and store the result
    initial_result = sh.basis

    # Change a property that affects the output of _compute_basis()
    sh.n_max = 3

    new_result = sh.basis

    # Call the method again and check that the result is different (cache miss)
    assert new_result is not initial_result

