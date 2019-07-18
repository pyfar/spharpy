import numpy as np
import scipy.special as _spspecial

from scipy.special import spherical_jn, sph_harm


def spherical_hankel(n, z, kind=2, derivative=False):
    if not derivative:
        if kind == 1:
            hankel = _spspecial.hankel1(n+0.5, z)
        elif kind == 2:
            hankel = _spspecial.hankel2(n+0.5, z)
        hankel = np.sqrt(np.pi/2/z) * hankel

    else:
        hankel = spherical_hankel(n-1, z, kind) - \
            (n+1)/z * spherical_hankel(n, z, kind)

    return hankel





# def spherical_hankel_prime(n, z, kind=2):
#     """Derivative of the spherical hankel function of second kind.
#     Not defined in the boost libs, therefore defined here."""
#     hankel_prime = spherical_hankel(n-1, z, kind) - \
#         (n+1)/z * spherical_hankel(n, z, kind)
#     return hankel_prime
