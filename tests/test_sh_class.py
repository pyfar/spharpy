"""
Tests for spherical harmonic class
"""
import pytest
import numpy as np
import pyfar as pf
from spharpy import SphericalHarmonics
from spharpy.samplings import gaussian, calculate_sampling_weights, equiangular

def test_sphharm_init():
    """Test default behaviour after initialization."""
    coordinates = equiangular(n_points=8)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    assert sph_harm.n_max == 2
    assert np.all(sph_harm.coordinates == coordinates)
    assert sph_harm.basis_type == 'real'
    assert sph_harm.normalization == 'n3d'
    assert sph_harm.channel_convention == 'acn'
    assert sph_harm.inverse_method == 'quadrature'
    assert sph_harm.condon_shortley == False

def test_sphharm_init_invalid_coordinates():
    with pytest.raises(TypeError,
                       match="coordinates must be a pyfar.Coordinates " \
                       "object or spharpy.SamplingSphere object"):
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
    coordinates = equiangular(n_points=8)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    assert sph_harm.basis_gradient_theta is not None
    assert sph_harm.basis_gradient_phi is not None

def test_sphharm_compute_inverse_quad():
    coordinates = gaussian(n_points=4)
    weights = calculate_sampling_weights(coordinates)
    coordinates.weights = weights
    sh = SphericalHarmonics(2, coordinates, inverse_method = 'quadrature')
    assert sh.basis_inv is not None

def test_sphharm_compute_inverse_pseudo_inv():
    coordinates = gaussian(n_points= 5)
    sh = SphericalHarmonics(2, coordinates,
                            inverse_method = 'pseudo_inverse')
    assert sh.basis_inv is not None

def test_compute_basis_caching():
    n_max = 2
    rng = np.random.default_rng()
    points = rng.integers(4, 10)
    coordinates = equiangular(n_points=points)
    sh = SphericalHarmonics(n_max, coordinates)

    # Call the method once and store the result
    initial_result = sh.basis

    # Change a property that affects the output of _compute_basis()
    sh.n_max = 3

    new_result = sh.basis

    # Call the method again and check that the result is different (cache miss)
    assert new_result is not initial_result

def test_setter_n_max():
    coordinates = equiangular(n_points=4)
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
    coordinates = equiangular(n_points=4)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    sph_harm.condon_shortley = "auto"
    assert sph_harm.condon_shortley == False

    with pytest.raises(TypeError):
        sph_harm.condon_shortley = 123  # Invalid type

def test_setter_channel_convention():
    coordinates = equiangular(n_points=4)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    sph_harm.channel_convention = "fuma"
    assert sph_harm.channel_convention == "fuma"

    with pytest.raises(ValueError):
        sph_harm.channel_convention = "invalid"  # Invalid value

    with pytest.raises(ValueError):
        sph_harm.n_max = 4
        sph_harm.channel_convention = "fuma"  # Invalid with n_max > 3

def test_setter_normalization():
    coordinates = equiangular(n_points=4)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    sph_harm.normalization = "sn3d"
    assert sph_harm.normalization == "sn3d"

    with pytest.raises(ValueError):
        sph_harm.normalization = "invalid"  # Invalid value

    with pytest.raises(ValueError):
        sph_harm.n_max = 4
        sph_harm.normalization = "maxN"  # Invalid with n_max > 3

def test_setter_inverse_method():
    coordinates = equiangular(n_points=4)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    sph_harm.inverse_method = "quadrature"
    assert sph_harm.inverse_method == "quadrature"

    with pytest.raises(ValueError,
                       match="Invalid inverse_method. Allowed: 'pseudo_inverse', " \
                       "'quadrature', or 'auto'."):
        sph_harm.inverse_method = "invalid"  # Invalid value
