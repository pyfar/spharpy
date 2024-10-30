import numpy.testing as npt
from pytest import raises, warns, mark
from spharpy.classes import SphericalHarmonicSignal
import numpy as np
import pyfar as pf
from spharpy import samplings
from spharpy.spherical import SphericalHarmonics


def test_spherical_harmonic_signal_init():
    """Test to init Signal without optional parameters."""

    signal = SphericalHarmonicSignal(np.array([1., 2., 3.]), 44100,
                                     n_max=1, basis_type='real',
                                     channel_convention='acn',
                                     phase_convention=True,
                                     normalization='n3d')
    assert isinstance(signal, SphericalHarmonicSignal)


def test_spherical_harmonic_signal_setter():
    pass


def test_spherical_harmonic__renormalize():
    pass


def test_spherical_harmonic__change_channel_convention():
    pass
