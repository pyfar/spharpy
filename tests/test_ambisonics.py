import numpy.testing as npt
from pytest import raises, warns, mark
from spharpy.classes import SphericalHarmonicSignal
import numpy as np


def test_spherical_harmonic_signal_init():
    """Test to init AmbisonicsSignal."""

    signal = SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.]]),
                                     44100, n_max=1, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d')
    assert isinstance(signal, SphericalHarmonicSignal)


def test_spherical_harmonic_signal_init_wrong_shape():
    """Test to init AmbisonicsSignal."""

    with raises(ValueError, match='Data has to few sh coefficients '
                'for n_max = {n_max}.'):
        SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                          [1., 2., 3.],
                                          [1., 2., 3.]]),
                                44100, n_max=1, basis_type='real',
                                channel_convention='acn',
                                normalization='n3d')


def test_spherical_harmonic_signal_normalization_setter():
    pass


def test_spherical_harmonic_signal_channel_convention_setter():
    pass


def test_spherical_harmonic__renormalize():
    pass


def test_spherical_harmonic__change_channel_convention():
    pass
