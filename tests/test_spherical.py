"""
Tests ...
"""
from pytest import raises
import numpy as np
import spharpy.spherical as sh


def test_renormalize():
    sh_data = np.ones((4, 2))
    current_norm = 'n3d'

    # test invalid type
    with raises(ValueError, match="Invalid normalization, has to be 'sn3d', "
                                  "'n3d', or 'maxN', but is wrong_norm"):
        sh.renormalize(sh_data, 'acn', current_norm, 'wrong_norm', axis=0)

    # test maxN
    sh_data_renorm = sh.renormalize(sh_data.copy(), 'acn', current_norm,
                                    'maxN', axis=0)
    sh_data_ref = np.array([[np.sqrt(1 / 2), np.sqrt(1 / 2)],
                            [np.sqrt(1 / 3), np.sqrt(1 / 3)],
                            [np.sqrt(1 / 3), np.sqrt(1 / 3)],
                            [np.sqrt(1 / 3), np.sqrt(1 / 3)]])

    np.testing.assert_equal(sh_data_renorm,
                            sh_data_ref)

    # test sn3d
    sh_data_renorm = sh.renormalize(sh_data.copy(), 'acn', current_norm,
                                    'sn3d', axis=0)
    sh_data_ref = np.array([[1 / np.sqrt(2 * 0 + 1), 1 / np.sqrt(2 * 0 + 1)],
                            [1 / np.sqrt(2 * 1 + 1), 1 / np.sqrt(2 * 1 + 1)],
                            [1 / np.sqrt(2 * 1 + 1), 1 / np.sqrt(2 * 1 + 1)],
                            [1 / np.sqrt(2 * 1 + 1), 1 / np.sqrt(2 * 1 + 1)]])

    np.testing.assert_equal(sh_data_renorm,
                            sh_data_ref)
    # test back to n3d
    sh_data_renorm = sh.renormalize(sh_data.copy(), 'acn', current_norm, 'n3d',
                                    axis=-2)
    np.testing.assert_equal(sh_data_renorm,
                            np.ones((4, 2)))


def test_change_channel_convention():
    sh_data = np.array([[1., 1., 1.],
                        [2., 2., 2.],
                        [3., 3., 3.],
                        [4., 4., 4.]])
    # test conversion to fuma                    
    current_channel_convention = 'acn'
    sh_data_new_convention = sh.change_channel_convention(
        sh_data, current_channel_convention, 'fuma', axis=0)
    sh_data_new_convention_fuma = sh_data[[0, 3, 1, 2], :]
    np.testing.assert_equal(sh_data_new_convention_fuma,
                            sh_data_new_convention)

    # test conversion to acn
    current_channel_convention = 'fuma'
    sh_data_new_convention = sh.change_channel_convention(
        sh_data_new_convention_fuma, current_channel_convention, 'acn', axis=0)
    np.testing.assert_equal(sh_data,
                            sh_data_new_convention)
