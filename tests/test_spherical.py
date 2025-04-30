"""
Tests ...
"""
from pytest import raises
import numpy as np
import spharpy.spherical as sh


def test_spherical_harmonic__renormalize():
    sh_data = np.ones((4, 8))
    current_norm = 'n3d'
    
    # test invalid type
    with raises(ValueError, match="Invalid normalization, has to be 'sn3d', "
                                  "'n3d', or 'maxN', but is wrong_norm"):
        sh.renormalize(sh_data, 'acn', current_norm, 'wrong_norm')

    # test maxN
    sh_data_renorm = sh.renormalize(sh_data, 'acn', current_norm, 'maxN',
                                    axis=-2)
    np.testing.assert_equal(sh_data_renorm,
                            np.array([np.sqrt(1 / 2),
                                      np.sqrt(1 / 3),
                                      np.sqrt(1 / 3),
                                      np.sqrt(1 / 3)]))

    # test sn3d
    sh_data_renorm = sh.renormalize(sh_data, 'acn', current_norm, 'sn3d',
                                    axis=-2)
    np.testing.assert_equal(sh_data_renorm,
                            np.array([1 / np.sqrt(2 * 0 + 1),
                                      1 / np.sqrt(2 * 1 + 1),
                                      1 / np.sqrt(2 * 1 + 1),
                                      1 / np.sqrt(2 * 1 + 1)]))
    # test back to n3d
    sh_data_renorm = sh.renormalize(sh_data, 'acn', current_norm, 'n3d',
                                    axis=-2)
    np.testing.assert_equal(sh_data_renorm,
                            np.ones((4, 8)))


def test_spherical_harmonic__change_channel_convention():
    sh_data = np.array([[1., 2., 3.],
                        [1., 2., 3.],
                        [1., 2., 3.],
                        [1., 2., 3.]])
    current_channel_convention = 'acn'

    sh_data_new_convention = sh.change_channel_convention(
        sh_data, current_channel_convention, 'fuma', axis=-2)

    sh_data_new_convention = sh.change_channel_convention(
        sh_data, current_channel_convention, 'acn', axis=-2)