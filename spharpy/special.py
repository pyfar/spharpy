"""
Subpackage implementing or wrapping special functions required in the
spharpy package.
"""

from itertools import count
import numpy as np
import scipy.special as _spspecial


def spherical_bessel(n, z, derivative=False):
    """
    Spherical bessel function of order n evaluated at z.

    .. math::

        j_n(z) = \\sqrt{\\frac{\\pi}{2z}} J_{n+\\frac{1}{2}} (z)

    Parameters
    ----------
    n : int, ndarray
        Order of the spherical bessel function
    z : double, ndarray
        Argument of the spherical bessel function. Has to be real valued.
    derivative : bool
        Return the derivative of the spherical Bessel function


    Returns
    -------
    jn : double, ndarray
        Spherical bessel function. Array with dimensions [N x Z], where N is
        the number of elements in n and Z is the number of elements in z.

    Note
    ----
    This is a wrapper around the Scipy implementation of the spherical Bessel
    function.

    """

    ufunc = _spspecial.spherical_jn
    n = np.asarray(n, dtype=np.int)
    z = np.asarray(z, dtype=np.double)

    bessel = np.zeros((n.size, z.size), dtype=np.complex)

    if n.size > 1:
        for idx, order in zip(count(), n):
            bessel[idx, :] = ufunc(order, z, derivative=derivative)
    else:
        bessel = ufunc(n, z, derivative=derivative)

    return bessel


def spherical_hankel(n, z, kind=2, derivative=False):
    """
    Spherical Hankel function of order n evaluated at z.

    .. math::

        j_n(z) = \\sqrt{\\frac{\\pi}{2z}} J_{n+\\frac{1}{2}} (z)

    Parameters
    ----------
    n : int, ndarray
        Order of the spherical bessel function
    z : double, ndarray
        Argument of the spherical bessel function. Has to be real valued.

    Returns
    -------
    hn : double, ndarray
        Spherical bessel function. Array with dimensions [N x Z], where N is
        the number of elements in n and Z is the number of elements in z.

    Note
    ----
    This is based on the Hankel functions implemented in the scipy package.
    """

    if kind not in (1, 2):
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


def sph_harm_real(n, m, theta, phi):
    """Real valued spherical harmonic function of order n and degree m evaluated
    at the angles theta and phi.

    Parameters
    ----------
    n : unsigned int
        The spherical harmonic order
    m : int
        The spherical harmonic degree
    theta : ndarray, double
        The elevation angle
    phi : ndarray, double
        The azimuth angle

    Returns
    -------
    Y_nm : ndarray, double
        The real valued spherial harmonic of order n and degree m

    """
    # careful here, scipy uses phi as the elevation angle and
    # theta as the azimuth angle
    Y_nm_cplx = _spspecial.sph_harm(m, n, phi, theta)

    if m == 0:
        Y_nm = np.real(Y_nm_cplx)
    elif m > 0:
        Y_nm = np.real(Y_nm_cplx) * np.sqrt(2)
    elif m < 0:
        Y_nm = np.imag(Y_nm_cplx) * np.sqrt(2) * np.float(-1)**(m+1)

    Y_nm *= (np.float(-1)**(m))

    return Y_nm
