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
    coordinates = pf.Coordinates.from_cartesian(coordinates.x, coordinates.y, coordinates.z)
    sh = SphericalHarmonics(2, coordinates, inverse_transform = 'quadrature')
    assert sh.basis_inv is not None

def test_sphharm_compute_inverse_pseudo_inv():
    coordinates = gaussian(n_points= 5)
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

def test_setter_n_max():
    coordinates = pf.Coordinates(1, 0, 0)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    sph_harm.n_max = 3
    assert sph_harm.n_max == 3

    with pytest.raises(ValueError):
        sph_harm.n_max = -1  # Invalid value

    with pytest.raises(ValueError):
        # set sph_harm to use 'fuma' channel convention
        sph_harm.n_max = 2  # Invalid with default 'acn' and 'maxN'
        sph_harm.channel_convention = "fuma"
        sph_harm.n_max = 4  # Invalid with default 'acn' and 'maxN'
    with pytest.raises(ValueError):
        sph_harm.channel_convention = "acn"
        sph_harm.n_max = 4
        # set maxN normalization
        sph_harm.normalization = "maxN"  # Invalid with n_max > 3
def test_setter_phase_convention():
    coordinates = pf.Coordinates(1, 0, 0)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    sph_harm.phase_convention = "auto"
    assert sph_harm.phase_convention == "auto"

    with pytest.raises(TypeError):
        sph_harm.phase_convention = 123  # Invalid type

def test_setter_weights():
    coordinates = gaussian(n_points=4)
    coordinates = pf.Coordinates.from_cartesian(coordinates.x, coordinates.y, coordinates.z)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    test_weights = np.array([1, 2, 3, 4])
    sph_harm.weights = test_weights
    assert np.array_equal(sph_harm.weights, test_weights)

def test_setter_channel_convention():
    coordinates = pf.Coordinates(1, 0, 0)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    sph_harm.channel_convention = "fuma"
    assert sph_harm.channel_convention == "fuma"

    with pytest.raises(ValueError):
        sph_harm.channel_convention = "invalid"  # Invalid value

    with pytest.raises(ValueError):
        sph_harm.n_max = 4
        sph_harm.channel_convention = "fuma"  # Invalid with n_max > 3

def test_setter_normalization():
    coordinates = pf.Coordinates(1, 0, 0)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    sph_harm.normalization = "sn3d"
    assert sph_harm.normalization == "sn3d"

    with pytest.raises(ValueError):
        sph_harm.normalization = "invalid"  # Invalid value

    with pytest.raises(ValueError):
        sph_harm.n_max = 4
        sph_harm.normalization = "maxN"  # Invalid with n_max > 3