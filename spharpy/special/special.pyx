# distutils: language = c++

# cython: embedsignature = True

"""
Extension module for special functions
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport sqrt, pow

import cython
from cython.parallel import prange

cdef extern from "boost/math/special_functions.hpp" namespace "boost::math":
    double sph_bessel(int n, double z) nogil;
    double sph_bessel_prime(int n, double z) nogil;
    complex sph_hankel_2(int n, double z) nogil;
    complex sph_hankel_1(int n, double z) nogil;
    double sph_neumann(int n, double z) nogil;
    double sph_neumann_prime(int n, double z) nogil;


cdef complex sph_hankel_2_prime(int n, double z) nogil:
    """Derivative of the spherical hankel function of second kind.
    Not defined in the boost libs, therefore defined here."""
    cdef complex hankel_2_prime
    hankel_2_prime = sph_hankel_2(n-1, z) - (n+1)/z * sph_hankel_2(n, z)
    return hankel_2_prime


cdef complex sph_hankel_1_prime(int n, double z) nogil:
    """Derivative of the spherical hankel function of first kind.
    Not defined in the boost libs, therefore defined here."""
    cdef complex hankel_1_prime
    hankel_1_prime = sph_hankel_1(n-1, z) - (n+1)/z * sph_hankel_1(n, z)
    return hankel_1_prime


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
                    sph_bessel(memview_order[idx_order], \
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
                    sph_bessel_prime(memview_order[idx_order], \
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
                        sph_hankel_1(memview_order[idx_order], \
                        memview_arg[idx_points])
            elif hankel_kind == 2:
                memview_hankel[idx_order, idx_points] = \
                        sph_hankel_2(memview_order[idx_order], \
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
                        sph_hankel_1_prime(memview_order[idx_order], \
                        memview_arg[idx_points])
            elif hankel_kind == 2:
                memview_hankel[idx_order, idx_points] = \
                        sph_hankel_2_prime(memview_order[idx_order], \
                        memview_arg[idx_points])

    return np.squeeze(hankel_prime)
