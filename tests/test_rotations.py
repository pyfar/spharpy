""" Tests for sh rotations"""

import pytest
import numpy as np
import spharpy.transforms as transforms
from spharpy.spherical import spherical_harmonic_basis

def test_rotation_matrix_z_axis_complex():
    rot_angle = np.pi/2
    n_max = 2
    reference = np.diag([1, 1j, 1, -1j, -1, 1j, 1, -1j, -1])

    rot_mat = transforms.rotation_z_axis(n_max, rot_angle)
    np.testing.assert_almost_equal(rot_mat, reference)

    rot_angle = np.pi
    reference = np.diag([1, -1, 1, -1, 1, -1, 1, -1, 1])

    rot_mat = transforms.rotation_z_axis(n_max, rot_angle)
    np.testing.assert_almost_equal(rot_mat, reference)

    rot_angle = 3/2*np.pi
    reference = np.diag([1, -1j, 1, 1j, -1, -1j, 1, 1j, -1])

    rot_mat = transforms.rotation_z_axis(n_max, rot_angle)
    np.testing.assert_almost_equal(rot_mat, reference)


def test_rotation_sh_basis_z_axis_complex():
    rot_angle = np.pi/2
    n_max = 2
    theta = np.asarray(np.pi/2)[np.newaxis]
    phi_x = np.asarray(0.0)[np.newaxis]
    phi_y = np.asarray(np.pi/2)[np.newaxis]
    reference = spherical_harmonic_basis(n_max, theta, phi_y).T.conj()

    rot_mat = transforms.rotation_z_axis(n_max, rot_angle)
    sh_vec_x = spherical_harmonic_basis(n_max, theta, phi_x)
    sh_vec_rotated = rot_mat @ sh_vec_x.T.conj()

    np.testing.assert_almost_equal(sh_vec_rotated, reference)
