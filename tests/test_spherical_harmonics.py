"""
Tests for spherical harmonic basis and related functions
"""

import sys
from filehandling import read_2d_matrix_from_csv

sys.path.append('./')

from unittest.mock import patch


import spharpy.spherical as sh
import spharpy.samplings as samplings
from spharpy.samplings import Coordinates
import numpy as np
import numpy.testing as npt


def test_spherical_harmonic():
    Nmax = 1
    theta = np.array([np.pi/2, np.pi/2, 0], dtype='double')
    phi = np.array([0, np.pi/2, 0], dtype='double')
    rad = np.ones(3, dtype=np.double)
    coords = Coordinates.from_spherical(rad, theta, phi)

    Y = np.array([[2.820947917738781e-01 + 0.000000000000000e+00j, 3.454941494713355e-01 + 0.000000000000000e+00j, 2.991827511286337e-17 + 0.000000000000000e+00j, -3.454941494713355e-01 + 0.000000000000000e+00j],
                  [2.820947917738781e-01 + 0.000000000000000e+00j, 2.115541521371041e-17 - 3.454941494713355e-01j, 2.991827511286337e-17 + 0.000000000000000e+00j, -2.115541521371041e-17 - 3.454941494713355e-01j],
                  [2.820947917738781e-01 + 0.000000000000000e+00j, 0.000000000000000e+00 + 0.000000000000000e+00j, 4.886025119029199e-01 + 0.000000000000000e+00j, 0.000000000000000e+00 + 0.000000000000000e+00j]], dtype=complex)


    basis = sh.spherical_harmonic_basis(Nmax, coords)

    np.testing.assert_allclose(Y, basis, atol=1e-13)


def test_spherical_harmonic_n10():
    Nmax = 10
    theta = np.array([np.pi/2, np.pi/2, 0], dtype='double')
    phi = np.array([0, np.pi/2, 0], dtype='double')
    n_points = len(theta)

    with patch.multiple(
            Coordinates,
            azimuth=phi,
            elevation=theta,
            n_points=n_points) as patched_vals:
        coords = Coordinates()

        Y = np.genfromtxt('./tests/data/sh_basis_cplx_n10.csv', delimiter=',', dtype=np.complex)
        basis = sh.spherical_harmonic_basis(Nmax, coords)

        np.testing.assert_allclose(Y, basis, atol=1e-13)


def test_spherical_harmonics_real():
    n_max = 10
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2], dtype='double')
    phi = np.array([0, np.pi/2, 0, np.pi/4], dtype='double')
    rad = np.ones(4)
    coords = Coordinates.from_spherical(rad, theta, phi)

    reference = read_2d_matrix_from_csv('./tests/data/sh_basis_real.csv')
    basis = sh.spherical_harmonic_basis_real(n_max, coords)
    np.testing.assert_allclose(basis, reference, atol=1e-13)


def test_orthogonality():
    """
    Check if the orthonormality condition of the spherical harmonics is fulfilled
    """
    n_max = 82
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2], dtype='double')
    phi = np.array([0, np.pi/2, 0, np.pi/4], dtype='double')
    n_points = phi.size
    n_points = len(theta)

    with patch.multiple(
            Coordinates,
            azimuth=phi,
            elevation=theta,
            n_points=n_points) as patched_vals:
        coords = Coordinates()
        basis = sh.spherical_harmonic_basis(n_max, coords)

        inner = (basis @ np.conjugate(basis.T))
        fact = 4 * np.pi / (n_max + 1) ** 2
        orth = np.diagonal(fact * inner)
        np.testing.assert_allclose(orth, np.ones(n_points), rtol=1e-15)


def test_orthogonality_real():
    """
    Check if the orthonormality condition of the reavl valued spherical harmonics is fulfilled
    """
    n_max = 82
    theta = np.array([np.pi / 2, np.pi / 2, 0, np.pi / 2], dtype='double')
    phi = np.array([0, np.pi / 2, 0, np.pi / 4], dtype='double')
    n_points = phi.size

    with patch.multiple(
            Coordinates,
            azimuth=phi,
            elevation=theta,
            n_points=n_points) as patched_vals:
        coords = Coordinates()
        basis = sh.spherical_harmonic_basis_real(n_max, coords)

        inner = (basis @ np.conjugate(basis.T))
        fact = 4 * np.pi / (n_max + 1) ** 2
        orth = np.diagonal(fact * inner)
        np.testing.assert_allclose(orth, np.ones(n_points), rtol=1e-15)


# def test_spherical_harmonic_derivative_phi():
#     n = 2
#     m = 1
#     sampling = samplings.equalarea(50, condition_num=np.inf)
#     Y_diff_phi = np.zeros((sampling.n_points), dtype=np.complex)
#     for idx in range(0, sampling.n_points):
#         Y_diff_phi[idx] = sh.spherical_harmonic_function_derivative_phi(
#                 n, m, sampling.elevation[idx], sampling.azimuth[idx])
#
#     ref_file = np.loadtxt('./tests/data/Y_diff_phi.csv', delimiter=',')
#     ref = ref_file[0] + 1j*ref_file[1]
#     npt.assert_allclose(ref, Y_diff_phi)


def test_spherical_harmonic_gradient_phi():
    n = 2
    m = 1
    sampling = samplings.equalarea(50, condition_num=np.inf)
    acn = sh.nm2acn(n, m)
    Y_grad_phi = sh.spherical_harmonic_basis_gradient(n, sampling)[1]

    ref_file = np.loadtxt('./tests/data/Y_grad_phi.csv', delimiter=',')
    ref = ref_file[0] + 1j*ref_file[1]
    npt.assert_allclose(ref, Y_grad_phi[:, acn])


def test_spherical_harmonic_derivative_theta():
    n = 2
    m = 1
    sampling = samplings.equalarea(50, condition_num=np.inf)
    acn = sh.nm2acn(n, m)
    Y_diff_theta = sh.spherical_harmonic_basis_gradient(n, sampling)[0]

    ref_file = np.loadtxt('./tests/data/Y_diff_theta.csv', delimiter=',')
    ref = ref_file[0] + 1j*ref_file[1]
    npt.assert_allclose(ref, Y_diff_theta[:, acn])


def test_spherical_harmonic_basis_gradient():
    n = 2
    m = 1
    sampling = samplings.equalarea(50, condition_num=np.inf)
    acn = sh.nm2acn(n, m)
    Y_diff_theta, Y_grad_phi = \
        sh.spherical_harmonic_basis_gradient(n, sampling)

    ref_file = np.loadtxt('./tests/data/Y_diff_theta.csv', delimiter=',')
    ref_theta = ref_file[0] + 1j*ref_file[1]
    npt.assert_allclose(ref_theta, Y_diff_theta[:, acn])

    ref_file = np.loadtxt('./tests/data/Y_grad_phi.csv', delimiter=',')
    ref_phi = ref_file[0] + 1j*ref_file[1]
    npt.assert_allclose(ref_phi, Y_grad_phi[:, acn])


# def test_spherical_harmonic_derivative_phi_real():
#     n = 5
#     m = 1
#     sampling = samplings.equalarea(50, condition_num=np.inf)
#     Y_diff_phi = np.zeros((sampling.n_points), dtype=np.complex)
#     for idx in range(0, sampling.n_points):
#         Y_diff_phi[idx] = sh.spherical_harmonic_function_derivative_phi(
#                 n, m, sampling.elevation[idx], sampling.azimuth[idx])
#
#     ref_file = np.loadtxt('./tests/data/Y_diff_phi.csv', delimiter=',')
#     ref = ref_file[0] + 1j*ref_file[1]
#     npt.assert_allclose(ref, Y_diff_phi)


def test_spherical_harmonic_gradient_phi_real():
    n = 5
    sampling = samplings.equalarea(30, condition_num=np.inf)
    Y_grad_phi = sh.spherical_harmonic_basis_gradient_real(n, sampling)[1]

    ref = np.loadtxt('./tests/data/Y_grad_phi_real.csv', delimiter=',')
    npt.assert_allclose(ref, Y_grad_phi, atol=1e-15)


def test_spherical_harmonic_derivative_theta_real():
    n = 5
    sampling = samplings.equalarea(30, condition_num=np.inf)
    Y_diff_theta = sh.spherical_harmonic_basis_gradient_real(n, sampling)[0]

    ref = np.loadtxt('./tests/data/Y_diff_theta_real.csv', delimiter=',')
    npt.assert_allclose(ref, Y_diff_theta, atol=1e-15)


def test_spherical_harmonic_basis_gradient_real():
    n = 5
    sampling = samplings.equalarea(30, condition_num=np.inf)
    Y_diff_theta, Y_grad_phi = \
        sh.spherical_harmonic_basis_gradient_real(n, sampling)

    ref_theta = np.loadtxt('./tests/data/Y_diff_theta_real.csv', delimiter=',')
    npt.assert_allclose(ref_theta, Y_diff_theta, atol=1e-15)

    ref_phi = np.loadtxt('./tests/data/Y_grad_phi_real.csv', delimiter=',')
    npt.assert_allclose(ref_phi, Y_grad_phi)
