"""
Tests for spherical harmonic basis and related functions
"""

import sys
from filehandling import read_2d_matrix_from_csv

sys.path.append('./')


import pytest
import spharpy.spherical as sh
import numpy as np

def test_spherical_harmonic():
    Nmax = 1
    theta = np.array([np.pi/2, np.pi/2, 0], dtype='double')
    phi = np.array([0, np.pi/2, 0], dtype='double')

    Y = np.array([[2.820947917738781e-01 + 0.000000000000000e+00j, 3.454941494713355e-01 + 0.000000000000000e+00j, 2.991827511286337e-17 + 0.000000000000000e+00j, -3.454941494713355e-01 + 0.000000000000000e+00j],
                  [2.820947917738781e-01 + 0.000000000000000e+00j, 2.115541521371041e-17 - 3.454941494713355e-01j, 2.991827511286337e-17 + 0.000000000000000e+00j, -2.115541521371041e-17 - 3.454941494713355e-01j],
                  [2.820947917738781e-01 + 0.000000000000000e+00j, 0.000000000000000e+00 + 0.000000000000000e+00j, 4.886025119029199e-01 + 0.000000000000000e+00j, 0.000000000000000e+00 + 0.000000000000000e+00j]], dtype=complex)


    basis = sh.spherical_harmonic_basis(Nmax, theta, phi)

    np.testing.assert_almost_equal(Y, basis)

def test_spherical_harmonics_real():
    n_max = 10
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2], dtype='double')
    phi = np.array([0, np.pi/2, 0, np.pi/4], dtype='double')

    reference = read_2d_matrix_from_csv('./tests/data/sh_basis_real.csv')
    basis = sh.spherical_harmonic_basis_real(n_max, theta, phi)
    np.testing.assert_almost_equal(reference,
                                   basis)

def test_orthogonality():
    """
    Check if the orthonormality condition of the spherical harmonics is fulfilled
    """
    n_max = 3
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2], dtype='double')
    phi = np.array([0, np.pi/2, 0, np.pi/4], dtype='double')

    for idx in range(0, theta.shape[0]):
        basis = sh.spherical_harmonic_basis(n_max, theta[idx][np.newaxis], phi[idx][np.newaxis])
        inner = (basis @ np.conjugate(basis.T))
        fact = 4*np.pi/(n_max+1)**2
        orth = fact * inner[0]
        np.testing.assert_almost_equal(1, orth)
