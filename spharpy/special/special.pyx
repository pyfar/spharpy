# distutils: language = c++

# cython: embedsignature = True

"""
Extension module for special functions
"""

import numpy as np
cimport numpy as cnp
import scipy.special as spspecial
from scipy.optimize import brentq

from libc.math cimport sqrt, pow

import cython
from cython.parallel import prange

cimport spharpy.special._special as _special

@cython.boundscheck(False)
@cython.wraparound(False)
def spherical_bessel(n, z):
    """
    Spherical bessel function of order n evaluated at z.

    .. math::

        j_n(z) = \\sqrt{\\frac{\\pi}{2z}} J_{n+\\frac{1}{2}} (z)

    References
    ----------
    .. [1]  https://www.boost.org/doc/libs/1_65_0/libs/math/doc/html/math_toolkit/bessel/sph_bessel.html

    Parameters
    ----------
    n : int, ndarray
        Order of the spherical bessel function

    z : double, ndarray
        Argument of the spherical bessel function. Has to be real valued.

    Returns
    -------
    jn : double, ndarray
        Spherical bessel function. Array with dimensions [N x Z], where N is
        the number of elements in n and Z is the number of elements in z.

    Note
    ----
    Implementation uses the implementation from the boost C++ libraries _[1].
    """
    cdef cnp.ndarray[long, ndim=1] order = \
            np.array(n, dtype=int, ndmin=1, copy=False)
    cdef cnp.ndarray[double, ndim=1] arg = \
            np.array(z, dtype=np.double, ndmin=1, copy=False)

    cdef long[::1] memview_order = order
    cdef double[::1] memview_arg = arg

    cdef Py_ssize_t n_coeff, n_points
    n_coeff = order.shape[0]
    n_points = arg.shape[0]

    cdef cnp.ndarray[double, ndim=2] bessel = \
            np.zeros((n_coeff, n_points), dtype=np.double)
    cdef double[:, ::1] memview_bessel = bessel

    cdef int idx_order, idx_points

    for idx_points in range(0, n_points):
        for idx_order in prange(0, n_coeff, nogil=True):
            memview_bessel[idx_order, idx_points] = \
                    _special.sph_bessel(memview_order[idx_order], \
                    memview_arg[idx_points])

    return np.squeeze(bessel)


@cython.boundscheck(False)
@cython.wraparound(False)
def spherical_bessel_derivative(n, z):
    """
    Derivative of the spherical bessel function of order n with respect to the
    argument z.

    Note
    ----
    Implementation from the boost C++ libraries _[2] used here, which uses the
    identity

    .. math::

        j_n^\\prime(z) = \\left(\\frac{n}{z}\\right) j_n(z) - j_{n+1}(z)

    References
    ----------
    .. [2]  https://www.boost.org/doc/libs/1_65_0/libs/math/doc/html/math_toolkit/bessel/sph_bessel.html

    Parameters
    ----------
    n : int, ndarray
        Order of the spherical bessel function

    z : double, ndarray
        Argument of the spherical bessel function. Has to be real valued.


    Returns
    -------
    jn_prime : double, ndarray
        Derivative of the spherical bessel function.
        Array with dimensions [N x Z], where N is the number of elements in n
        and Z is the number of elements in z.
    """
    cdef cnp.ndarray[long, ndim=1] order = \
            np.array(n, dtype=int, ndmin=1, copy=False)
    cdef cnp.ndarray[double, ndim=1] arg = \
            np.array(z, dtype=np.double, ndmin=1, copy=False)

    cdef long[::1] memview_order = order
    cdef double[::1] memview_arg = arg

    cdef Py_ssize_t n_coeff, n_points
    n_coeff = order.shape[0]
    n_points = arg.shape[0]

    cdef cnp.ndarray[double, ndim=2] bessel_prime = \
            np.zeros((n_coeff, n_points), dtype=np.double)
    cdef double[:, ::1] memview_bessel = bessel_prime

    cdef int idx_order, idx_points

    for idx_points in range(0, n_points):
        for idx_order in prange(0, n_coeff, nogil=True):
            memview_bessel[idx_order, idx_points] = \
                    _special.sph_bessel_prime(memview_order[idx_order], \
                    memview_arg[idx_points])

    return np.squeeze(bessel_prime)



@cython.boundscheck(False)
@cython.wraparound(False)
def spherical_hankel(n, z, kind=2):
    """
    Spherical hankel function of order n evaluated at z.

    .. math::

        j_n(z) = \\sqrt{\\frac{\\pi}{2z}} J_{n+\\frac{1}{2}} (z)

    References
    ----------
    .. [1]  https://www.boost.org/doc/libs/1_65_0/libs/math/doc/html/math_toolkit/bessel/sph_bessel.html

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
    Implementation uses the implementation from the boost C++ libraries _[1].
    """
    cdef cnp.ndarray[long, ndim=1] order = \
            np.array(n, dtype=int, ndmin=1, copy=False)
    cdef cnp.ndarray[double, ndim=1] arg = \
            np.array(z, dtype=np.double, ndmin=1, copy=False)

    cdef long[::1] memview_order = order
    cdef double[::1] memview_arg = arg

    kind = int(kind)
    if kind != 1 and kind != 2:
        raise ValueError("The spherical hankel function can only be of first or second kind.")
    cdef int hankel_kind = kind

    cdef Py_ssize_t n_coeff, n_points
    n_coeff = order.shape[0]
    n_points = arg.shape[0]

    cdef cnp.ndarray[complex, ndim=2] hankel = \
            np.zeros((n_coeff, n_points), dtype=np.complex)
    cdef complex[:, ::1] memview_hankel = hankel

    cdef int idx_order, idx_points

    for idx_points in range(0, n_points):
        for idx_order in prange(0, n_coeff, nogil=True):
            if hankel_kind == 1:
                memview_hankel[idx_order, idx_points] = \
                        _special.sph_hankel_1(memview_order[idx_order], \
                        memview_arg[idx_points])
            elif hankel_kind == 2:
                memview_hankel[idx_order, idx_points] = \
                        _special.sph_hankel_2(memview_order[idx_order], \
                        memview_arg[idx_points])

    return np.squeeze(hankel)


@cython.boundscheck(False)
@cython.wraparound(False)
def spherical_hankel_derivative(n, z, kind=2):
    """
    Derivative of the spherical bessel function of order n with respect to the
    argument z.

    Note
    ----
    Implementation from the boost C++ libraries _[2] used here, which uses the
    identity

    .. math::

        j_n^\\prime(z) = \\left(\\frac{n}{z}\\right) j_n(z) - j_{n+1}(z)

    References
    ----------
    .. [2]  https://www.boost.org/doc/libs/1_65_0/libs/math/doc/html/math_toolkit/bessel/sph_bessel.html

    Parameters
    ----------
    n : int, ndarray
        Order of the spherical bessel function

    z : double, ndarray
        Argument of the spherical bessel function. Has to be real valued.


    Returns
    -------
    jn_prime : double, ndarray
        Derivative of the spherical bessel function.
        Array with dimensions [N x Z], where N is the number of elements in n
        and Z is the number of elements in z.
    """
    cdef cnp.ndarray[long, ndim=1] order = \
            np.array(n, dtype=int, ndmin=1, copy=False)
    cdef cnp.ndarray[double, ndim=1] arg = \
            np.array(z, dtype=np.double, ndmin=1, copy=False)

    cdef long[::1] memview_order = order
    cdef double[::1] memview_arg = arg

    kind = int(kind)
    if kind != 1 and kind != 2:
        raise ValueError("The spherical hankel function can only be of first or second kind.")
    cdef int hankel_kind = kind

    cdef Py_ssize_t n_coeff, n_points
    n_coeff = order.shape[0]
    n_points = arg.shape[0]

    cdef cnp.ndarray[complex, ndim=2] hankel_prime = \
            np.zeros((n_coeff, n_points), dtype=np.complex)
    cdef complex[:, ::1] memview_hankel = hankel_prime

    cdef int idx_order, idx_points

    for idx_points in range(0, n_points):
        for idx_order in prange(0, n_coeff, nogil=True):
            if hankel_kind == 1:
                memview_hankel[idx_order, idx_points] = \
                        _special.sph_hankel_1_prime(memview_order[idx_order], \
                        memview_arg[idx_points])
            elif hankel_kind == 2:
                memview_hankel[idx_order, idx_points] = \
                        _special.sph_hankel_2_prime(memview_order[idx_order], \
                        memview_arg[idx_points])

    return np.squeeze(hankel_prime)


def spherical_bessel_zeros(n_max, n_zeros):
    """Compute the zeros of the spherical bessel function.
    This function will always start at order zero which is equal
    to sin(x)/x and iteratively compute the roots for higher orders.
    The roots are computed using Brents algorith from scipy.

    Parameters
    ----------
    n_max : int
        The order of the spherical bessel function
    n_zeros : int
        The number of roots to be computed

    Returns
    -------
    roots : ndarray, double
        The roots of the spherical bessel function

    """
    def func(x, n):
        return spspecial.spherical_jn(n, x)

    zerosj = np.zeros((n_max+1, n_zeros), dtype=np.float)
    zerosj[0] = np.arange(1, n_zeros+1)*np.pi
    points = np.arange(1, n_zeros+n_max+1)*np.pi

    roots = np.zeros(n_zeros+n_max, dtype=np.float)
    for i in range(1,n_max+1):
        for j in range(n_zeros+n_max-i):
            roots[j] = brentq(func, points[j], points[j+1], (i,), maxiter=5000)
        points = roots
        zerosj[i, :n_zeros] = roots[:n_zeros]

    return zerosj
