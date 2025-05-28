from pytest import raises
from spharpy.classes import SphericalHarmonicSignal
import numpy as np


def test_spherical_harmonic_signal_init():
    """Test init SphercalHarmonicsSignal."""

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
    """Test if Condon-Shortley is set properly in init
       SphercalHarmonicsSignal."""

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


def test_spherical_harmonic_signal_init_multichannel():
    """Test init SphercalHarmonicsSignal."""
    sh_coeffs = np.zeros((2, 6, 16))
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
