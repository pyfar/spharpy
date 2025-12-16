"""
Tests for spherical harmonic class.
"""
import pytest
import numpy as np
from spharpy import (SphericalHarmonics, SamplingSphere,
                     SphericalHarmonicDefinition)
import deepdiff


def test_spherical_harmonics_definition_init():
    """Test default behavior."""
    definition = SphericalHarmonicDefinition()
    assert definition.basis_type == 'real'
    assert definition.normalization == 'N3D'
    assert definition.channel_convention == 'ACN'
    assert definition.condon_shortley is False


@pytest.mark.parametrize("phase_convention", [True, False])
def test_setter_phase_convention(phase_convention):
    """Test setting Condon-Shortley phase convention with boolean values."""
    sph_harm = SphericalHarmonicDefinition()
    sph_harm.condon_shortley = phase_convention
    assert sph_harm.condon_shortley == phase_convention


def test_init_phase_convention_auto():
    """Test initialization with auto Condon-Shortley phase convention."""
    sph_harm = SphericalHarmonicDefinition(
        basis_type="complex",
        condon_shortley="auto")
    assert sph_harm.condon_shortley is True

    sph_harm = SphericalHarmonicDefinition(
        basis_type="real",
        condon_shortley="auto")
    assert sph_harm.condon_shortley is False

def test_setter_phase_convention_auto():
    """Test setting Condon-Shortley phase convention to auto."""
    sph_harm = SphericalHarmonicDefinition()
    sph_harm.basis_type = "complex"
    sph_harm.condon_shortley = "auto"
    assert sph_harm.condon_shortley is True

    sph_harm = SphericalHarmonicDefinition()
    sph_harm.basis_type = "real"
    sph_harm.condon_shortley = "auto"
    assert sph_harm.condon_shortley is False


def test_setter_phase_convention_invalid():
    """Test error handling for invalid Condon-Shortley phase values."""
    sph_harm = SphericalHarmonicDefinition()
    with pytest.raises(ValueError, match='must be a bool or the string'):
        sph_harm.condon_shortley = 123  # Invalid type

    with pytest.raises(ValueError, match='must be a bool or the string'):
        sph_harm.condon_shortley = "invalid"  # Invalid string


@pytest.mark.parametrize("channel_convention", ["ACN", "FuMa"])
def test_setter_channel_convention_definition(channel_convention):
    """Test setting channel convention for spherical harmonic definition."""
    sph_harm = SphericalHarmonicDefinition()
    sph_harm.channel_convention = channel_convention
    assert sph_harm.channel_convention == channel_convention


@pytest.mark.parametrize("channel_convention", ["ACN", "FuMa"])
def test_init_channel_convention_definition(channel_convention):
    """Test initialization with different channel conventions."""
    sph_harm = SphericalHarmonicDefinition(
        channel_convention=channel_convention)
    assert sph_harm.channel_convention == channel_convention


def test_setter_channel_convention_fuma_error():
    """Test error when setting FUMA channel convention with n_max > 3."""
    sph_harm = SphericalHarmonicDefinition(n_max=4)

    message = 'n_max > 3 is not allowed with'

    with pytest.raises(ValueError, match=message):
        sph_harm.channel_convention = "FuMa"

    with pytest.raises(ValueError, match=message):
        SphericalHarmonicDefinition(
            n_max=4,
            channel_convention="FuMa",
        )


def test_setter_channel_convention_definition_invalid():
    """Test error handling for invalid channel convention values."""
    sph_harm = SphericalHarmonicDefinition()
    with pytest.raises(ValueError, match='Invalid channel convention'):
        sph_harm.channel_convention = "invalid"  # Invalid value


@pytest.mark.parametrize("normalization", ["N3D", "SN3D", "maxN", "NM", "SNM"])
def test_setter_normalization_definition(normalization):
    """Test setting different normalization conventions."""
    sph_harm = SphericalHarmonicDefinition()
    sph_harm.normalization = normalization
    assert sph_harm.normalization == normalization


@pytest.mark.parametrize("normalization", ["N3D", "SN3D", "maxN", "NM", "SNM"])
def test_init_normalization_definition(normalization):
    """Test initialization with different normalization conventions."""
    sph_harm = SphericalHarmonicDefinition(
        normalization=normalization)
    assert sph_harm.normalization == normalization


def test_setter_normalization_definition_invalid():
    """Test error handling for invalid normalization values."""
    sph_harm = SphericalHarmonicDefinition()
    with pytest.raises(ValueError, match='Invalid normalization'):
        sph_harm.normalization = "invalid"  # Invalid value


def test_setter_normalization():
    """Test error when setting maxN normalization with n_max > 3."""
    sph_harm = SphericalHarmonicDefinition(n_max=4)

    with pytest.raises(ValueError, match='n_max > 3 is not allowed with'):
        sph_harm.normalization = "maxN"  # Invalid with n_max > 3


def test_sh_definition_setter_n_max():
    """Test setting n_max property and error handling for invalid values."""
    sph_harm = SphericalHarmonicDefinition(n_max=2)
    sph_harm.n_max = 3
    assert sph_harm.n_max == 3

    with pytest.raises(ValueError, match='n_max must be a positive integer'):
        sph_harm.n_max = -1  # Invalid value


@pytest.mark.parametrize(
        ('norm', 'convention'), [('maxN', 'ACN'), ('N3D', 'FuMa')])
def test_sh_definition_setter_n_max_invalid_combinations(norm, convention):
    """Test error when setting n_max > 3 with incompatible combinations."""
    sph_harm = SphericalHarmonicDefinition(n_max=2)

    sph_harm.channel_convention = convention
    sph_harm.normalization = norm
    with pytest.raises(ValueError, match='n_max > 3 is not allowed'):
        sph_harm.n_max = 4


def test_setter_basis_type():
    """Test setting basis type and error handling for invalid values."""
    sph_harm = SphericalHarmonicDefinition()
    sph_harm.basis_type = "complex"
    assert sph_harm.basis_type == "complex"

    sph_harm.basis_type = "real"
    assert sph_harm.basis_type == "real"

    with pytest.raises(ValueError, match='Invalid basis type'):
        sph_harm.basis_type = "invalid"  # Invalid value


def test_sphharm_init(icosahedron_sampling):
    """Test default behaviour after initialization."""
    sph_harm = SphericalHarmonics(n_max=2, coordinates=icosahedron_sampling)
    assert sph_harm.n_max == 2
    assert np.all(sph_harm.coordinates == icosahedron_sampling)
    assert sph_harm.inverse_method == 'quadrature'

def test_sphharm_init_invalid_coordinates():
    """Test error handling for invalid coordinate types."""
    with pytest.raises(TypeError,
                       match="coordinates must be a pyfar.Coordinates " \
                       "object or spharpy.SamplingSphere object"):
        SphericalHarmonics(n_max=2, coordinates=[0, 0, 1])

def test_sphharm_init_invalid_n_max(icosahedron_sampling):
    """Test error handling for invalid n_max values."""
    with pytest.raises(ValueError, match='n_max must be a positive integer'):
        SphericalHarmonics(n_max=-1, coordinates=icosahedron_sampling)

def test_sphharm_init_from_definition():
    """Test initialization using SphericalHarmonicDefinition."""

    n_max = 0
    coordinates = SamplingSphere(1, 0, 0)
    inverse_method = "auto"

    # generate from definition
    definition = SphericalHarmonicDefinition(n_max)
    sh_from_definition = SphericalHarmonics.from_definition(
        coordinates, definition, inverse_method)

    # generate manually
    sh_manually = SphericalHarmonics(
        n_max, coordinates, definition.basis_type, definition.normalization,
        definition.channel_convention, inverse_method,
        definition.condon_shortley)

    assert not deepdiff.DeepDiff(
        sh_from_definition.__dict__, sh_manually.__dict__)

def test_sphharm_init_from_definition_error():
    """Test error when passing wrong type to `from_definition` class method."""

    message = "definition must be a SphericalHarmonicDefinition"

    with pytest.raises(TypeError, match=message):
        SphericalHarmonics.from_definition(
            SamplingSphere(1, 0, 0), "definition")

def test_sphharm_compute_basis(icosahedron_sampling):
    """Test spherical harmonic basis computation."""
    sph_harm = SphericalHarmonics(n_max=2, coordinates=icosahedron_sampling)
    assert sph_harm.basis is not None

def test_sphharm_compute_basis_gradient(icosahedron_sampling):
    """Test spherical harmonic basis gradient computation."""
    sph_harm = SphericalHarmonics(n_max=2, coordinates=icosahedron_sampling)
    assert sph_harm.basis_gradient_theta is not None
    assert sph_harm.basis_gradient_phi is not None

def test_sphharm_compute_inverse_quad(icosahedron_sampling):
    """Test spherical harmonic inverse computation using quadrature method."""
    sh = SphericalHarmonics(
        1, coordinates=icosahedron_sampling, inverse_method='quadrature')
    assert sh.basis_inv is not None

def test_sphharm_compute_inverse_pseudo_inv(icosahedron_sampling):
    """Test spherical harmonic inverse using pseudo-inverse method."""
    sh = SphericalHarmonics(2, coordinates=icosahedron_sampling,
                            inverse_method='pseudo_inverse')
    assert sh.basis_inv is not None

def test_compute_basis_caching(icosahedron_sampling):
    """Test that basis computation results are cached and invalidated."""
    n_max = 1
    sh = SphericalHarmonics(
        n_max, icosahedron_sampling,
        basis_type='real',
        normalization='N3D',
        channel_convention='ACN',
        condon_shortley=False,
    )

    # Call the method once and store the result
    last_result = sh.basis

    sh.n_max = n_max  # Setting to the same value should not reset cache
    # Call the method again and check that the result is the same (cache hit)
    assert sh.basis is last_result

    # Change a property that affects the output of _compute_basis()
    sh.n_max = 3

    new_result = sh.basis

    # Call the method again and check that the result is different (cache miss)
    assert new_result is not last_result
    last_result = new_result

    sh.normalization = 'SN3D'
    new_result = sh.basis
    assert new_result is not last_result
    last_result = new_result

    sh.channel_convention = 'FuMa'
    new_result = sh.basis
    assert new_result is not last_result
    last_result = new_result

    sh.basis_type = 'complex'
    new_result = sh.basis
    assert new_result is not last_result
    last_result = new_result

    sh.condon_shortley = True
    new_result = sh.basis
    assert new_result is not last_result


def test_setter_inverse_method(icosahedron_sampling):
    """Test setting inverse method and error handling for invalid values."""
    sph_harm = SphericalHarmonics(n_max=2, coordinates=icosahedron_sampling)
    sph_harm.inverse_method = "quadrature"
    assert sph_harm.inverse_method == "quadrature"

    with pytest.raises(
            ValueError,
            match=("Invalid inverse_method. Allowed: 'pseudo_inverse', "
                   "'quadrature', or 'auto'.")):
        sph_harm.inverse_method = "invalid"  # Invalid value
