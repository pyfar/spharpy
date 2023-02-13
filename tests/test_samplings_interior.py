import numpy as np
import spharpy.samplings as samplings
from spharpy.special import spherical_bessel_zeros
from scipy.special import spherical_jn


def test_sph_bessel_zeros():
    roots = spherical_bessel_zeros(3, 3)
    jn_zeros = np.ones(roots.shape)
    for n in range(4):
        jn_zeros[n, :] = spherical_jn(n, roots[n, :])
    zeros = np.zeros((4, 3), dtype=float)
    np.testing.assert_allclose(jn_zeros, zeros, atol=1e-12)


def test_interior_points_chardon():
    kr_max = 7
    int_points = samplings.interior_stabilization_points(kr_max)

    filename = 'tests/data/interior_points_kr7.csv'
    truth = np.genfromtxt(filename, dtype=float, delimiter=';')

    np.testing.assert_allclose(int_points.cartesian.T, truth, atol=1e-7)
