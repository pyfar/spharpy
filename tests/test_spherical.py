"""
Tests renormalization and change channel convention methods
"""
from pytest import raises, mark
import numpy as np
import spharpy.spherical as sh


def test_renormalize_errors():
    sh_data = np.ones((4, 2))

    # test channel convention
    with raises(ValueError,
                match="Invalid channel convention. Has to be 'acn' "
                      "or 'fuma', but is wrong_channel_convention"):
        sh.renormalize(sh_data, 'wrong_channel_convention', 'maxN',
                       'n3d', axis=0)

    # test current norm
    with raises(ValueError, match="Invalid normalization. Has to be 'sn3d', "
                                  "'n3d', or 'maxN', but is wrong_norm"):
        sh.renormalize(sh_data, 'acn', 'wrong_norm', 'n3d', axis=0)

    # test target norm
    with raises(ValueError, match="Invalid normalization. Has to be 'sn3d', "
                                  "'n3d', or 'maxN', but is wrong_norm"):
        sh.renormalize(sh_data, 'acn', 'n3d', 'wrong_norm', axis=0)


@mark.parametrize("channel_convention", ['acn', 'fuma'])
def test_renormalize(channel_convention):
    sh_data = np.ones((4, 2))

    # test from n3d to maxN
    current_norm = 'n3d'
    target_norm = 'maxN'
    sh_data_n3d_to_maxN = sh.renormalize(sh_data, channel_convention,
                                         current_norm,
                                         target_norm, axis=0)
    sh_data_ref = np.array([[np.sqrt(1 / 2), np.sqrt(1 / 2)],
                            [np.sqrt(1 / 3), np.sqrt(1 / 3)],
                            [np.sqrt(1 / 3), np.sqrt(1 / 3)],
                            [np.sqrt(1 / 3), np.sqrt(1 / 3)]])

    np.testing.assert_equal(sh_data_n3d_to_maxN,
                            sh_data_ref)

    # test from maxN to n3d
    current_norm = 'maxN'
    target_norm = 'n3d'
    sh_data_maxN_to_n3d = sh.renormalize(sh_data_n3d_to_maxN,
                                         channel_convention,
                                         current_norm,
                                         target_norm, axis=0)

    np.testing.assert_equal(sh_data_maxN_to_n3d,
                            np.ones((4, 2)))

    # test from maxN to sn3d
    current_norm = 'maxN'
    target_norm = 'sn3d'
    sh_data_maxN_to_sn3d = sh.renormalize(sh_data_n3d_to_maxN,
                                          channel_convention,
                                          current_norm,
                                          target_norm, axis=0)
    # back to n3d to check against 0
    sh_data_sn3d_to_n3d = sh.renormalize(sh_data_maxN_to_sn3d,
                                         channel_convention,
                                         'sn3d',
                                         'n3d', axis=0)
    np.testing.assert_equal(sh_data_sn3d_to_n3d,
                            np.ones((4, 2)))

    # test from n3d to sn3d
    current_norm = 'n3d'
    target_norm = 'sn3d'
    sh_data_n3d_to_sn3d = sh.renormalize(sh_data, channel_convention,
                                         current_norm,
                                         target_norm, axis=0)
    sh_data_ref = np.array([[1 / np.sqrt(2 * 0 + 1), 1 / np.sqrt(2 * 0 + 1)],
                            [1 / np.sqrt(2 * 1 + 1), 1 / np.sqrt(2 * 1 + 1)],
                            [1 / np.sqrt(2 * 1 + 1), 1 / np.sqrt(2 * 1 + 1)],
                            [1 / np.sqrt(2 * 1 + 1), 1 / np.sqrt(2 * 1 + 1)]])

    np.testing.assert_equal(sh_data_n3d_to_sn3d,
                            sh_data_ref)

    # test from sn3d to n3d
    current_norm = 'sn3d'
    target_norm = 'n3d'
    sh_data_sn3d_to_n3d = sh.renormalize(sh_data_n3d_to_sn3d,
                                         channel_convention,
                                         current_norm,
                                         target_norm,
                                         axis=-2)
    np.testing.assert_equal(sh_data_sn3d_to_n3d,
                            np.ones((4, 2)))

    # test from sn3d to maxN
    current_norm = 'sn3d'
    target_norm = 'maxN'
    sh_data_sn3d_to_maxN = sh.renormalize(sh_data_n3d_to_sn3d,
                                          channel_convention,
                                          current_norm,
                                          target_norm, axis=0)
    # back to n3d to check against 0
    sh_data_maxN_to_n3d = sh.renormalize(sh_data_sn3d_to_maxN,
                                         channel_convention,
                                         'maxN',
                                         'n3d', axis=0)
    np.testing.assert_equal(sh_data_maxN_to_n3d,
                            np.ones((4, 2)))


def test_change_channel_convention_errors():
    sh_data = np.ones((4, 2))
    # test current channel convention
    with raises(ValueError, match="Invalid current channel convention. Has to "
                                  "be 'acn' or 'fuma', but is wrong"):
        sh.change_channel_convention(sh_data, 'wrong', 'fuma', axis=0)

    # test target channel convention
    with raises(ValueError, match="Invalid target channel convention. Has to "
                                  "be 'acn' or 'fuma', but is wrong"):
        sh.change_channel_convention(sh_data, 'fuma', 'wrong', axis=0)


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
