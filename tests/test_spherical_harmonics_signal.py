from pytest import raises
from spharpy.classes import SphericalHarmonicSignal
import numpy as np
import re


def test_spherical_harmonic_signal_init():
    """Test init SphericalHarmonicsSignal."""

    data = np.array([[1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]]).reshape(1, 4, 3)
    signal = SphericalHarmonicSignal(data,
                                     44100, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d',
                                     condon_shortley=False)
    assert isinstance(signal, SphericalHarmonicSignal)


def test_spherical_harmonic_signal_init_condon_shortley():
    """
    Test if Condon-Shortley is set properly in init
    SphericalHarmonicsSignal.
    """

    data = np.array([[1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]]).reshape(1, 4, 3)
    signal = SphericalHarmonicSignal(data,
                                     44100, basis_type='real',
                                     channel_convention='acn',
                                     condon_shortley=False,
                                     normalization='n3d')
    assert not signal.condon_shortley

    signal = SphericalHarmonicSignal(data.astype(np.complex128),
                                     44100, basis_type='complex',
                                     channel_convention='acn',
                                     normalization='n3d',
                                     condon_shortley=True,
                                     is_complex=True)
    assert signal.condon_shortley


def test_spherical_harmonic_signal_wrong_dimensions():
    """Test dimensions of SH coefficient data."""

    data = np.array([[1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]])

    # test if dimension of data is < 3
    with raises(ValueError,
                match="Invalid number of dimensions. Data should have "
                      "at least 3 dimensions."):
        SphericalHarmonicSignal(data,
                                44100, basis_type='real',
                                channel_convention='acn',
                                condon_shortley=False,
                                normalization='n3d')
    # test if sh channels are valid
    data = np.array([[1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]]).reshape(1, 5, 3)

    with raises(ValueError,
                match=re.escape("Invalid number of SH channels: "
                                f"{data.shape[-2]}. It must match "
                                "(n_max + 1)^2.")):
        SphericalHarmonicSignal(data,
                                44100, basis_type='real',
                                channel_convention='acn',
                                condon_shortley=False,
                                normalization='n3d')


def test_spherical_harmonic_signal_init_multichannel():
    """Test init SphercalHarmonicsSignal."""
    sh_coeffs = np.zeros((2, 4, 16))
    signal = SphericalHarmonicSignal(sh_coeffs,
                                     44100, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d',
                                     condon_shortley=False)
    assert isinstance(signal, SphericalHarmonicSignal)


def test_nmax_getter():
    """Test nmax getter."""
    data = np.array([[1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]]).reshape(1, 4, 3)
    signal = SphericalHarmonicSignal(data,
                                     44100, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d',
                                     condon_shortley=False)
    assert signal.n_max == 1
    assert isinstance(signal.n_max, int)


def test_init_wrong_basis_type():
    data = np.array([[1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]]).reshape(1, 4, 3)
    with raises(ValueError,
                match="Invalid basis type, only "
                      "'complex' and 'real' are supported"):
        SphericalHarmonicSignal(data,
                                44100, basis_type='invalid_basis_type',
                                channel_convention='acn',
                                normalization='n3d',
                                condon_shortley=False)


def test_basis_type_getter():
    """Test basis_type getter."""
    data = np.array([[1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]]).reshape(1, 4, 3)
    signal = SphericalHarmonicSignal(data,
                                     44100, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d',
                                     condon_shortley=False)
    assert signal.basis_type == 'real'


def test_init_wrong_normalization():
    data = np.array([[1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]]).reshape(1, 4, 3)

    with raises(ValueError,
                match="Invalid normalization, has to be 'sn3d', "
                      "'n3d', or 'maxN', but is invalid_normalization"):
        SphericalHarmonicSignal(data,
                                44100, basis_type='real',
                                channel_convention='acn',
                                normalization='invalid_normalization',
                                condon_shortley=False)


def test_spherical_harmonic_signal_normalization_setter():
    data = np.array([[1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]]).reshape(1, 4, 3)
    signal = SphericalHarmonicSignal(data,
                                     44100, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d',
                                     condon_shortley=False)
    signal.normalization = 'sn3d'
    assert signal.normalization == 'sn3d'

    signal.normalization = 'maxN'
    assert signal.normalization == 'maxN'

    signal.normalization = 'n3d'
    assert signal.normalization == 'n3d'


def test_init_wrong_channel_convention():
    data = np.array([[1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]]).reshape(1, 4, 3)

    with raises(ValueError,
                match="Invalid channel convention, has to be 'acn' "
                      "or 'fuma', but is invalid_convention"):
        SphericalHarmonicSignal(data,
                                44100, basis_type='real',
                                channel_convention='invalid_convention',
                                normalization='n3d',
                                condon_shortley=False)


def test_spherical_harmonic_signal_channel_convention_setter():
    data = np.array([[1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]]).reshape(1, 4, 3)
    signal = SphericalHarmonicSignal(data,
                                     44100, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d',
                                     condon_shortley=False)
    signal.channel_convention = 'fuma'
    assert signal.channel_convention == 'fuma'

    signal.channel_convention = 'acn'
    assert signal.channel_convention == 'acn'
