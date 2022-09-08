"""
Tests for the aperture function of a spherical loudspeaker array
"""
import spharpy.spherical as sh
from scipy.special import legendre
from spharpy.indexing import sph_identity_matrix
import numpy as np


def test_aperture_diag():
    n_max = 10
    rad_cap = 0.1
    rad_sphere = 1
    aperture = sh.aperture_vibrating_spherical_cap(n_max, rad_sphere, rad_cap)

    angle_cap = np.arcsin(rad_cap / rad_sphere)
    arg = np.cos(angle_cap)
    n_sh = n_max+1
    reference = np.zeros(n_sh, dtype=float)
    reference[0] = (1-arg)*2*np.pi
    for n in range(1, n_max+1):
        legendre_minus = legendre(n-1)(arg)
        legendre_plus = legendre(n+1)(arg)
        legendre_term = legendre_minus - legendre_plus
        reference[n] = legendre_term * 4 * np.pi / (2*n+1)

    np.testing.assert_allclose(
        aperture,
        np.diag(sph_identity_matrix(n_max, 'n-nm').T @ reference))
