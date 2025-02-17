import numpy.testing as npt
from pytest import raises, warns, mark
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


def test_spherical_harmonic_signal_init_wrong_nmax():
    """Test init SphercalHarmonicsSignal."""

    with raises(ValueError, match="n_max must be an integer value"):
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
    """Test to init AmbisonicsSignal."""

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
    """Test nmax getter."""
    signal = SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.]]),
                                     44100, n_max=1, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d')
    assert signal.basis_type == 'real'


def test_spherical_harmonic_signal_normalization_setter():
    raise NotImplementedError()


def test_spherical_harmonic_signal_channel_convention_setter():
    raise NotImplementedError()


def test_spherical_harmonic__renormalize():
    sh_signal = SphericalHarmonicSignal(np.ones((4, 8)),
                                        44100, n_max=1, basis_type='real',
                                        channel_convention='acn',
                                        normalization='n3d')
    # test invalid type
    with raises(ValueError, match="Invalid normalization, has to be 'sn3d', "
                                  "'n3d', or 'maxN, but is wrong_norm"):
        sh_signal._renormalize('wrong_norm')

    # test maxN
    sh_signal_ = sh_signal.copy()
    sh_signal_._renormalize('maxN')

    # test sn3d
    sh_signal_ = sh_signal.copy()
    sh_signal_._renormalize('sn3d')


def test_spherical_harmonic__change_channel_convention():
    raise NotImplementedError()
