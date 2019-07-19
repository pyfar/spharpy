import numpy as np
import scipy.special as _spspecial
from itertools import count


def spherical_bessel(n, z, derivative=False):
    ufunc = _spspecial.spherical_jn
    n = np.asarray(n, dtype=np.int)
    z = np.asarray(z, dtype=np.double)

    bessel = np.zeros((n.size, z.size), dtype=np.complex)

    if n.size > 1:
        for idx, order in zip(count(), n):
            bessel[idx, :] = ufunc(order, z, derivative=derivative)
    else:
        bessel = ufunc(order, z, derivative=derivative)

    return bessel


def spherical_hankel(n, z, kind=2, derivative=False):
    if kind not in (1,2):
        raise ValueError("The spherical hankel function can \
            only be of first or second kind.")

    n = np.asarray(n, dtype=np.int)
    z = np.asarray(z, dtype=np.double)

    if derivative:
        ufunc = _spherical_hankel_derivative
    else:
        ufunc = _spherical_hankel

    if n.size > 1:
        hankel = np.zeros((n.size, z.size), dtype=np.complex)
        for idx, order in zip(count(), n):
            hankel[idx, :] = ufunc(order, z, kind)
    else:
        hankel = ufunc(n, z, kind)

    return hankel


def _spherical_hankel(n, z, kind):
    if kind == 1:
        hankel = _spspecial.hankel1(n+0.5, z)
    elif kind == 2:
        hankel = _spspecial.hankel2(n+0.5, z)
    hankel = np.sqrt(np.pi/2/z) * hankel

    return hankel


def _spherical_hankel_derivative(n, z, kind):
    hankel = _spherical_hankel(n-1, z, kind) - \
        (n+1)/z * _spherical_hankel(n, z, kind)

    return hankel
