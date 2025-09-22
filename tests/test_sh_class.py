"""
Tests for spherical harmonic class.
"""
import pytest
import numpy as np
import pyfar as pf
from spharpy import SphericalHarmonics
from spharpy.samplings import gaussian, calculate_sampling_weights, equiangular
from spharpy.classes.sh import SphericalHarmonicDefinition


def test_spherical_harmonics_definition_init():
    """Test default behavior.
    """
    definition = SphericalHarmonicDefinition()
    assert definition.basis_type == 'real'
    assert definition.normalization == 'n3d'
    assert definition.channel_convention == 'acn'
    assert definition.condon_shortley is False


@pytest.mark.parametrize("phase_convention", [True, False])
def test_setter_phase_convention(phase_convention):
    sph_harm = SphericalHarmonicDefinition()
    sph_harm.condon_shortley = phase_convention
    assert sph_harm.condon_shortley == phase_convention


def test_init_phase_convention_auto():
    sph_harm = SphericalHarmonicDefinition(
        basis_type="complex",
        condon_shortley="auto")
    assert sph_harm.condon_shortley is True

    sph_harm = SphericalHarmonicDefinition(
        basis_type="real",
        condon_shortley="auto")
    assert sph_harm.condon_shortley is False

def test_setter_phase_convention_auto():
    sph_harm = SphericalHarmonicDefinition()
    sph_harm.basis_type = "complex"
    sph_harm.condon_shortley = "auto"
    assert sph_harm.condon_shortley is True

    sph_harm = SphericalHarmonicDefinition()
    sph_harm.basis_type = "real"
    sph_harm.condon_shortley = "auto"
    assert sph_harm.condon_shortley is False


def test_setter_phase_convention_invalid():
    sph_harm = SphericalHarmonicDefinition()
    with pytest.raises(ValueError, match='must be a bool or the string'):
        sph_harm.condon_shortley = 123  # Invalid type

    with pytest.raises(ValueError, match='must be a bool or the string'):
        sph_harm.condon_shortley = "invalid"  # Invalid string


@pytest.mark.parametrize("channel_convention", ["acn", "fuma"])
def test_setter_channel_convention_definition(channel_convention):
    sph_harm = SphericalHarmonicDefinition()
    sph_harm.channel_convention = channel_convention
    assert sph_harm.channel_convention == channel_convention


@pytest.mark.parametrize("channel_convention", ["acn", "fuma"])
def test_init_channel_convention_definition(channel_convention):
    sph_harm = SphericalHarmonicDefinition(
        channel_convention=channel_convention)
    assert sph_harm.channel_convention == channel_convention



def test_setter_channel_convention_definition_invalid():
    sph_harm = SphericalHarmonicDefinition()
    with pytest.raises(ValueError, match='Invalid channel convention'):
        sph_harm.channel_convention = "invalid"  # Invalid value


@pytest.mark.parametrize("normalization", ["n3d", "sn3d", "maxN"])
def test_setter_normalization_definition(normalization):
    sph_harm = SphericalHarmonicDefinition()
    sph_harm.normalization = normalization
    assert sph_harm.normalization == normalization


@pytest.mark.parametrize("normalization", ["n3d", "sn3d", "maxN"])
def test_init_normalization_definition(normalization):
    sph_harm = SphericalHarmonicDefinition(
        normalization=normalization)
    assert sph_harm.normalization == normalization


def test_setter_basis_type():
    sph_harm = SphericalHarmonicDefinition()
    sph_harm.basis_type = "complex"
    assert sph_harm.basis_type == "complex"

    sph_harm.basis_type = "real"
    assert sph_harm.basis_type == "real"

    with pytest.raises(ValueError, match='Invalid basis type'):
        sph_harm.basis_type = "invalid"  # Invalid value


def test_setter_normalization_definition_invalid():
    sph_harm = SphericalHarmonicDefinition()
    with pytest.raises(ValueError, match='Invalid normalization'):
        sph_harm.normalization = "invalid"  # Invalid value


def test_sphharm_init():
    """Test default behaviour after initialization."""
    coordinates = equiangular(n_points=8)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    assert sph_harm.n_max == 2
    assert np.all(sph_harm.coordinates == coordinates)
    assert sph_harm.inverse_method == 'quadrature'

def test_sphharm_init_invalid_coordinates():
    with pytest.raises(TypeError,
                       match="coordinates must be a pyfar.Coordinates " \
                       "object or spharpy.SamplingSphere object"):
        SphericalHarmonics(n_max=2, coordinates=[0, 0, 1])

def test_sphharm_init_invalid_n_max():
    coordinates = pf.Coordinates(1, 0, 0)
    with pytest.raises(ValueError, match='n_max must be a positive integer'):
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
    sh = SphericalHarmonics(2, coordinates, inverse_method='quadrature')
    assert sh.basis_inv is not None

def test_sphharm_compute_inverse_pseudo_inv():
    coordinates = gaussian(n_points= 5)
    sh = SphericalHarmonics(2, coordinates,
                            inverse_method='pseudo_inverse')
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

    with pytest.raises(ValueError, match='n_max must be a positive integer'):
        sph_harm.n_max = -1  # Invalid value

    # set sph_harm to use 'fuma' channel convention
    sph_harm.n_max = 2  # Invalid with default 'acn' and 'maxN'
    sph_harm.channel_convention = "fuma"
    with pytest.raises(ValueError, match='n_max > 3 is not allowed'):
        sph_harm.n_max = 4  # Invalid with default 'acn' and 'maxN'

    sph_harm.channel_convention = "acn"
    sph_harm.n_max = 4
    with pytest.raises(ValueError, match='n_max > 3 is not allowed'):
        # set maxN normalization
        sph_harm.normalization = "maxN"  # Invalid with n_max > 3


def test_setter_channel_convention():
    coordinates = equiangular(n_points=4)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)

    sph_harm.channel_convention = "fuma"  # Invalid with n_max > 3
    with pytest.raises(ValueError, match='n_max > 3 is not allowed with'):
        sph_harm.n_max = 4

def test_setter_normalization():
    coordinates = equiangular(n_points=4)
    sph_harm = SphericalHarmonics(n_max=4, coordinates=coordinates)

    with pytest.raises(ValueError, match='n_max > 3 is not allowed with'):
        sph_harm.normalization = "maxN"  # Invalid with n_max > 3

def test_setter_inverse_method():
    coordinates = equiangular(n_points=4)
    sph_harm = SphericalHarmonics(n_max=2, coordinates=coordinates)
    sph_harm.inverse_method = "quadrature"
    assert sph_harm.inverse_method == "quadrature"

    with pytest.raises(
            ValueError,
            match=("Invalid inverse_method. Allowed: 'pseudo_inverse', "
                   "'quadrature', or 'auto'.")):
        sph_harm.inverse_method = "invalid"  # Invalid value
