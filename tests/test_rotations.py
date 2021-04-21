""" Tests for sh rotations"""

import numpy as np
import spharpy.transforms as transforms
from spharpy.spherical import spherical_harmonic_basis
from spharpy.samplings import Coordinates
import spharpy
from spharpy.transforms import RotationSH
import pytest


def test_rotation_matrix_z_axis_complex():
    rot_angle = np.pi/2
    n_max = 2
    reference = np.diag([1, 1j, 1, -1j, -1, 1j, 1, -1j, -1])

    rot_mat = transforms.rotation_z_axis(n_max, rot_angle)
    np.testing.assert_allclose(rot_mat, reference)

    rot_angle = np.pi
    reference = np.diag([1, -1, 1, -1, 1, -1, 1, -1, 1])

    rot_mat = transforms.rotation_z_axis(n_max, rot_angle)
    np.testing.assert_allclose(rot_mat, reference)

    rot_angle = 3/2*np.pi
    reference = np.diag([1, -1j, 1, 1j, -1, -1j, 1, 1j, -1])

    rot_mat = transforms.rotation_z_axis(n_max, rot_angle)
    np.testing.assert_allclose(rot_mat, reference)


def test_rotation_sh_basis_z_axis_complex():
    rot_angle = np.pi/2
    n_max = 2
    coords_x = Coordinates(1, 0, 0)
    coords_y = Coordinates(0, 1, 0)
    reference = spherical_harmonic_basis(n_max, coords_y).T.conj()

    rot_mat = transforms.rotation_z_axis(n_max, rot_angle)
    sh_vec_x = spherical_harmonic_basis(n_max, coords_x)
    sh_vec_rotated = rot_mat @ sh_vec_x.T.conj()

    sampling = spharpy.samplings.dodecahedron()
    Y = spharpy.spherical.spherical_harmonic_basis(n_max, sampling)

    np.testing.assert_allclose(Y@sh_vec_rotated, Y@reference)


def test_rotation_maxtrix_z_axis_real():
    n_max = 2
    rot_angle = np.pi/4
    reference = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, np.cos(rot_angle), 0, np.sin(rot_angle), 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, -np.sin(rot_angle), 0, np.cos(rot_angle), 0, 0, 0, 0, 0],
        [0, 0, 0, 0, np.cos(2*rot_angle), 0, 0, 0, np.sin(2*rot_angle)],
        [0, 0, 0, 0, 0, np.cos(rot_angle), 0, np.sin(rot_angle), 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, -np.sin(rot_angle), 0, np.cos(rot_angle), 0],
        [0, 0, 0, 0, -np.sin(2*rot_angle), 0, 0, 0, np.cos(2*rot_angle)]])
    rot_matrix = transforms.rotation_z_axis_real(n_max, rot_angle)
    np.testing.assert_allclose(rot_matrix, reference)


def test_rotation_sh_basis_z_axis_real():
    rot_angle = np.pi/2
    n_max = 2
    coords_x = Coordinates(1, 0, 0)
    coords_y = Coordinates(0, 1, 0)
    reference = np.squeeze(
        spharpy.spherical.spherical_harmonic_basis_real(n_max, coords_y))

    rot_mat = spharpy.transforms.rotation_z_axis_real(n_max, rot_angle)
    sh_vec_x = spharpy.spherical.spherical_harmonic_basis_real(n_max, coords_x)
    sh_vec_rotated = np.squeeze(rot_mat @ np.squeeze(sh_vec_x))

    np.testing.assert_almost_equal(sh_vec_rotated, np.squeeze(reference))


def test_wigner_d_rot_real():
    euler_angles = []
    n_max = 5
    euler_angles = np.deg2rad(np.array([90., 45., -90.]))
    D = spharpy.transforms.wigner_d_rotation_real(
        n_max, euler_angles[0], euler_angles[1], euler_angles[2])

    reference = np.squeeze(spharpy.spherical.spherical_harmonic_basis_real(
            n_max, Coordinates(0, 1, 1)))

    sh_vec = np.squeeze(spharpy.spherical.spherical_harmonic_basis_real(
        n_max, Coordinates(0, 1, 0)))
    sh_vec_rotated = D @ sh_vec

    np.testing.assert_allclose(sh_vec_rotated, reference, atol=1e-10)

    rot_angle_z = np.pi/4
    D = spharpy.transforms.wigner_d_rotation_real(
        n_max, rot_angle_z, 0, 0)

    rot_mat = spharpy.transforms.rotation_z_axis_real(n_max, rot_angle_z)

    np.testing.assert_allclose(D, rot_mat, atol=1e-7)

    pass


def test_wigner_d_rot():
    euler_angles = []
    n_max = 5
    euler_angles = np.deg2rad(np.array([90., 45., -90.]))
    D = spharpy.transforms.wigner_d_rotation(
        n_max, euler_angles[0], euler_angles[1], euler_angles[2])

    reference = np.squeeze(spharpy.spherical.spherical_harmonic_basis(
            n_max, Coordinates(0, 1, 1)))

    sh_vec = np.squeeze(spharpy.spherical.spherical_harmonic_basis(
        n_max, Coordinates(0, 1, 0)))
    sh_vec_rotated = D @ sh_vec

    np.testing.assert_allclose(sh_vec_rotated, reference)

    rot_angle_z = np.pi/4
    D = spharpy.transforms.wigner_d_rotation(
        n_max, rot_angle_z, 0, 0)

    rot_mat = spharpy.transforms.rotation_z_axis(n_max, rot_angle_z)

    np.testing.assert_allclose(D, rot_mat, atol=1e-7)


def test_RotationSH():
    n_max = 4
    rot_angle_z = np.pi/2
    rot_vec = [0, 0, rot_angle_z]
    rot = RotationSH.from_rotvec(n_max, rot_vec)

    assert rot._n_max == n_max

    n_max = 2
    rot.n_max = n_max
    assert rot._n_max == n_max
    assert rot.n_max == n_max

    with pytest.raises(ValueError):
        rot.n_max = -1

    D_Rot = rot.as_spherical_harmonic(type='real')

    reference = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, np.cos(rot_angle_z), 0, np.sin(rot_angle_z), 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, -np.sin(rot_angle_z), 0, np.cos(rot_angle_z), 0, 0, 0, 0, 0],
        [0, 0, 0, 0, np.cos(2*rot_angle_z), 0, 0, 0, np.sin(2*rot_angle_z)],
        [0, 0, 0, 0, 0, np.cos(rot_angle_z), 0, np.sin(rot_angle_z), 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, -np.sin(rot_angle_z), 0, np.cos(rot_angle_z), 0],
        [0, 0, 0, 0, -np.sin(2*rot_angle_z), 0, 0, 0, np.cos(2*rot_angle_z)]])

    np.testing.assert_allclose(D_Rot, reference, atol=1e-10)

    rot = RotationSH.from_rotvec(n_max, [0, 0, 90], degrees=True)
    np.testing.assert_allclose(
        rot.as_spherical_harmonic(type='real'),
        reference, atol=1e-10)

    rot = RotationSH.from_euler(n_max, 'zyz', [0, 0, 90], degrees=True)
    np.testing.assert_allclose(
        rot.as_spherical_harmonic(type='real'),
        reference, atol=1e-10)

    rot = RotationSH.from_quat(n_max, [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)])
    np.testing.assert_allclose(
        rot.as_spherical_harmonic(type='real'),
        reference, atol=1e-10)

    rot_mat_z_spat = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]])
    rot = RotationSH.from_matrix(n_max, rot_mat_z_spat)
    np.testing.assert_allclose(
        rot.as_spherical_harmonic(type='real'),
        reference, atol=1e-7)

    # mrp = [0, 0, np.tan(np.pi/2/4)]
    # rot = RotationSH.from_mrp(n_max, mrp)
    # np.testing.assert_allclose(
    #     rot.as_spherical_harmonic(type='real'),
    #     reference, atol=1e-7)
