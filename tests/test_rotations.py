"""Tests for sh rotations."""

import numpy as np
import spharpy.transforms as transforms
from spharpy.spherical import spherical_harmonic_basis
import spharpy
from spharpy.transforms import SphericalHarmonicRotation
from spharpy.classes.sh import (
    SphericalHarmonicDefinition, SphericalHarmonics)
import pytest
from pyfar import Coordinates
from spharpy.classes.audio import SphericalHarmonicSignal
import pyfar


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


def test_SphericalHarmonicRotation():
    """Test SphericalHarmonicRotation creation and check against reference
    for a 90 deg rotation around the z-axis.
    """
    n_max = 2
    definition = SphericalHarmonicDefinition(n_max=n_max)
    rot_angle_z = np.pi/2
    rot_vec = [0, 0, rot_angle_z]
    rot = SphericalHarmonicRotation.from_rotvec(rot_vec)

    D_Rot = rot.as_spherical_harmonic_matrix(definition)

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

    rot = SphericalHarmonicRotation.from_rotvec([0, 0, 90], degrees=True)
    np.testing.assert_allclose(
        rot.as_spherical_harmonic_matrix(definition),
        reference, atol=1e-10)

    rot = SphericalHarmonicRotation.from_euler('zyz', [0, 0, 90], degrees=True)
    np.testing.assert_allclose(
        rot.as_spherical_harmonic_matrix(definition),
        reference, atol=1e-10)

    rot = SphericalHarmonicRotation.from_quat(
        [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)])
    np.testing.assert_allclose(
        rot.as_spherical_harmonic_matrix(definition),
        reference, atol=1e-10)

    rot_mat_z_spat = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]])
    rot = SphericalHarmonicRotation.from_matrix(rot_mat_z_spat)
    np.testing.assert_allclose(
        rot.as_spherical_harmonic_matrix(definition),
        reference, atol=1e-7)


def test_SphericalHarmonicRotation_apply():
    """
    Test application of SphericalHarmonicRotation to SphericalHarmonicSignal
    using the apply method, the multiplication operator, and the rotation
    matrix.
    """
    n_max = 2
    spherical_harmonics = SphericalHarmonics(
        n_max=n_max, coordinates=Coordinates(1, 0, 0), inverse_method=None)
    rot_angle_z = np.pi/2
    rot_vec = [0, 0, rot_angle_z]
    rot = SphericalHarmonicRotation.from_rotvec(rot_vec)

    noise = pyfar.signals.noise(512)

    sh_signal = SphericalHarmonicSignal(
        np.atleast_3d(
            spherical_harmonics.basis.T * noise.time).transpose(1, 0, 2),
        noise.sampling_rate,
        basis_type=spherical_harmonics.basis_type,
        normalization=spherical_harmonics.normalization,
        channel_convention=spherical_harmonics.channel_convention,
        condon_shortley=spherical_harmonics.condon_shortley,
    )

    sh_signal_rotated = rot.apply(sh_signal)

    rot_mat = rot.as_spherical_harmonic_matrix(sh_signal)

    sh_data_rotated = rot_mat @ sh_signal._data

    np.testing.assert_allclose(
        sh_signal_rotated._data,
        sh_data_rotated,
        atol=1e-10)

    sh_signal_rot_operator = rot * sh_signal

    np.testing.assert_allclose(
        sh_signal_rotated._data,
        sh_signal_rot_operator._data,
        atol=1e-10)


def test_SphericalHarmonicRotation_mul_rotations():
    """Test multiplication of two SphericalHarmonicRotation objects."""
    rot_angle_z = np.pi/2
    rot_vec = [0, 0, rot_angle_z]
    rot = SphericalHarmonicRotation.from_rotvec(rot_vec)

    result = rot * rot
    result_matrix = result.as_matrix()

    ref = rot.as_matrix() @ rot.as_matrix()

    np.testing.assert_allclose(
        result_matrix,
        ref,
        atol=1e-10)


def test_SphericalHarmonicRotation_mul_invalid():
    """Test if invalid multiplication operations raise errors."""
    rot = SphericalHarmonicRotation.from_rotvec([0, 0, np.pi/2])

    with pytest.raises(ValueError, match="Multiplication is only supported"):
        rot * 42 # type: ignore


def test_SphericalHarmonicRotation_invalid_definition():
    """Test if invalid spherical harmonic definitions raise errors."""
    rot = SphericalHarmonicRotation.from_rotvec([0, 0, np.pi/2])

    definition = SphericalHarmonicDefinition(
        n_max=2, normalization='N3D', channel_convention='FuMa')
    with pytest.raises(
            NotImplementedError,
            match="Only 'ACN' channel convention is supported",
        ):
        rot.as_spherical_harmonic_matrix(definition)

    definition = SphericalHarmonicDefinition(
        n_max=2, normalization='SN3D', channel_convention='ACN')
    with pytest.raises(
            NotImplementedError,
            match="Only 'N3D' normalization is supported",
        ):
        rot.as_spherical_harmonic_matrix(definition)

