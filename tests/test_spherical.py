"""
Tests ...
"""
from pytest import raises
import numpy as np
import spharpy.spherical as sh


def test_spherical_harmonic__renormalize():
    sh_signal = SphericalHarmonicSignal(np.ones((4, 8)),
                                        44100, n_max=1, basis_type='real',
                                        channel_convention='acn',
                                        normalization='n3d')
    # test invalid type
    with raises(ValueError, match="Invalid normalization, has to be 'sn3d', "
                                  "'n3d', or 'maxN', but is wrong_norm"):
        sh_signal._renormalize('wrong_norm')

    # test maxN
    sh_signal_ = sh_signal.copy()
    sh.renormalize('maxN')
    np.testing.assert_equal(sh_signal_.time[:, 0],
                            np.array([np.sqrt(1 / 2),
                                      np.sqrt(1 / 3),
                                      np.sqrt(1 / 3),
                                      np.sqrt(1 / 3)]))

    # test sn3d
    sh_signal_ = sh_signal.copy()
    sh_signal_._renormalize('sn3d')
    np.testing.assert_equal(sh_signal_.time[:, 0],
                            np.array([1 / np.sqrt(2 * 0 + 1),
                                      1 / np.sqrt(2 * 1 + 1),
                                      1 / np.sqrt(2 * 1 + 1),
                                      1 / np.sqrt(2 * 1 + 1)]))
    # test back to n3d
    sh.renormalize('n3d')
    np.testing.assert_equal(sh_signal_.time,
                            np.ones((4, 8)))


def test_spherical_harmonic__change_channel_convention():
    signal = SphericalHarmonicSignal(np.array([[1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.],
                                               [1., 2., 3.]]),
                                     44100, n_max=1, basis_type='real',
                                     channel_convention='acn',
                                     normalization='n3d')
    sh.change_channel_convention()
    assert signal.channel_convention == 'fuma'

    signal._change_channel_convention()
    assert signal.channel_convention == 'acn'
