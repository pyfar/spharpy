from pytest import raises
from spharpy.classes import SphericalHarmonicSignal
import numpy as np


def test_spherical_harmonic_signal_init():
    """Test init SphercalHarmonicsSignal."""

    signal = SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.]]),
                                     44100, n_max=1, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d')
    assert isinstance(signal, SphericalHarmonicSignal)


def test_spherical_harmonic_signal_init_condon_shortley():
    """Test if condon shortley is set properly in init
       SphercalHarmonicsSignal."""

    signal = SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.]]),
                                     44100, n_max=1, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d')
    assert not signal.condon_shortley

    signal = SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.]], dtype=complex),
                                     44100, n_max=1, basis_type='complex',
                                     channel_convention='acn',
                                     normalization='n3d',
                                     is_complex=True)
    assert signal.condon_shortley


def test_spherical_harmonic_signal_init_wrong_nmax():
    """Test init SphericalHarmonicsSignal."""

    with raises(ValueError, match="n_max must be a positive integer"):
        SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                          [1., 2., 3.],
                                          [1., 2., 3.]]),
                                44100, n_max=1.2, basis_type='real',
                                channel_convention='acn',
                                normalization='n3d')

    with raises(ValueError, match="n_max must be a positive integer"):
        SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                          [1., 2., 3.],
                                          [1., 2., 3.]]),
                                44100, n_max=-1, basis_type='real',
                                channel_convention='acn',
                                normalization='n3d')


def test_spherical_harmonic_signal_init_multichannel():
    """Test init SphercalHarmonicsSignal."""
    sh_coeffs = np.zeros((2, 6, 16))
    signal = SphericalHarmonicSignal(sh_coeffs,
                                     44100, n_max=1, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d')
    assert isinstance(signal, SphericalHarmonicSignal)


def test_spherical_harmonic_signal_init_wrong_shape():
    """Test to init SphericalHarmonicsSignal."""

    with raises(ValueError, match='Data has to few sh coefficients '
                'for n_max=1. Highest possible n_max is 0'):
        SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                          [1., 2., 3.],
                                          [1., 2., 3.]]),
                                44100, n_max=1, basis_type='real',
                                channel_convention='acn',
                                normalization='n3d')


def test_nmax_getter():
    """Test nmax getter."""
    signal = SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.]]),
                                     44100, n_max=1, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d')
    assert signal.n_max == 1


def test_basis_type_getter():
    """Test basis_type getter."""
    signal = SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.]]),
                                     44100, n_max=1, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d')
    assert signal.basis_type == 'real'


def test_spherical_harmonic_signal_normalization_setter():
    signal = SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.]]),
                                     44100, n_max=1, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d')
    signal.normalization = 'sn3d'
    assert signal.normalization == 'sn3d'

    signal.normalization = 'maxN'
    assert signal.normalization == 'maxN'

    signal.normalization = 'n3d'
    assert signal.normalization == 'n3d'


def test_spherical_harmonic_signal_channel_convention_setter():
    signal = SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.]]),
                                     44100, n_max=1, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d')
    signal.channel_convention = 'fuma'
    assert signal.channel_convention == 'fuma'

    signal.channel_convention = 'acn'
    assert signal.channel_convention == 'acn'
