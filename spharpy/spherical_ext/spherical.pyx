# distutils: language = c++

# cython: embedsignature = True
"""
Spherical extension module docstring
"""


import numpy as np
cimport numpy as cnp

from libc.stdlib cimport free
from libc.math cimport ceil, sqrt, pow

import cython
from cython.parallel import prange

from spharpy.samplings import Coordinates

cdef extern from "boost/math/special_functions/spherical_harmonic.hpp" namespace "boost::math":
    double complex spherical_harmonic(unsigned order, int degree, double theta, double phi) nogil;
    double spherical_harmonic_r(unsigned order, int degree, double theta, double phi) nogil;
    double spherical_harmonic_i(unsigned order, int degree, double theta, double phi) nogil;
cdef extern from "bessel_functions.h":
    complex *make_modal_strength(unsigned n_max, double *kr, int n_bins, int arraytype);
cdef extern from "special_functions.h":
    double legendre_polynomial(int n, double x);
    complex sph_hankel_2(int n, double z);
    complex sph_hankel_2_prime(int n, double z);


cdef class _finalizer:
    """
    Finalizer class that frees memory after array is deleted in python.
    This is a helper function that is only available inside of Cython.
    """
    cdef void *_data
    def __dealloc__(self):
        if self._data is not NULL:
            free(self._data)


cdef void set_base(cnp.ndarray arr, void *carr):
    """
    Set base for underlying memory of numpy arrays
    The class _finalizer is used to free memory after array is deleted
    This is a helper function that is only available inside of Cython.
    """
    cdef _finalizer f = _finalizer()
    f._data = <void*>carr
    cnp.set_array_base(arr, f)


def spherical_harmonic_derivative_phi(n, m, theta, phi):
    """Calculate the derivative of the spherical hamonics with respect to
    the azimuth angle phi.

    Parameters
    ----------

    n : int
        Spherical harmonic order
    m : int
        Spherical harmonic degree
    theta : double
        Elevation angle 0 < theta < pi
    phi : double
        Azimuth angle 0 < phi < 2*pi

    Returns
    -------

    sh_diff : complex double
        Spherical harmonic derivative

    """
    if m == 0:
        res = 0.0
    else:
        res = spherical_harmonic_function(n, m, theta, phi) * 1j * m

    return res


def spherical_harmonic_derivative_theta(n, m, theta, phi):
    """Calculate the derivative of the spherical hamonics with respect to
    the elevation angle theta.

    Parameters
    ----------

    n : int
        Spherical harmonic order
    m : int
        Spherical harmonic degree
    theta : double
        Elevation angle 0 < theta < pi
    phi : double
        Azimuth angle 0 < phi < 2*pi

    Returns
    -------

    sh_diff : complex double
        Spherical harmonic derivative

    Note
    ----

    This implementation is subject to singularities at the poles due to the
    1/sin(theta) term.

    """
    if n == 0:
        res = 0.0
    else:
        first = spherical_harmonic_function(n, m, theta, phi) * (n+1)
        second = spherical_harmonic_function(n+1, m, theta, phi) \
                * np.sqrt(n**2-m**2+2*n+1) * np.sqrt(2*n+1) / np.sqrt(2*n + 3)

        res = (-first/np.tan(theta) + second/np.sin(theta))

    return res


cdef complex spherical_harmonic_function(unsigned n, int m, double theta, double phi) nogil:
    """Simple wrapper function for the boost spherical harmonic function."""
    return spherical_harmonic(n, m, theta, phi)


cdef double spherical_harmonic_function_real(unsigned n, int m, double theta, double phi) nogil:
    """Use c math library here for speed and numerical robustness.
    Using the numpy ** operator instead of the libc pow function yields
    numeric issues which result in sign errors."""

    cdef double Y_nm = 0.0
    if (m == 0):
        Y_nm = spherical_harmonic_r(n, m, theta, phi)
    elif (m > 0):
        Y_nm = spherical_harmonic_r(n, m, theta, phi) * sqrt(2)
    elif (m < 0):
        Y_nm = spherical_harmonic_i(n, m, theta, phi) * sqrt(2) * <double>pow(-1, m+1)

    return Y_nm * <double>pow(-1, m)


def nm2acn(n, m):
    """
    Calculate the linear index coefficient for a spherical harmonic order n
    and degree m, according to the Ambisonics Channel Convention [1]_.

    .. math::

        acn = n^2 + n + m

    References
    ----------
    .. [1]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A Suggested Ambisonics
            Format (revised by F. Zotter),” International Symposium on Ambisonics and Spherical
            Acoustics, vol. 3, pp. 1–11, 2011.


    Parameters
    ----------
    n : integer, ndarray
        Spherical harmonic order
    m : integer, ndarray
        Spherical harmonic degree

    Returns
    -------
    acn : integer, ndarray
        Linear index

    """
    n = np.asarray(n, dtype=np.int)
    m = np.asarray(m, dtype=np.int)
    n_acn = m.size

    if not (n.size == m.size):
        raise ValueError("n and m need to be of the same size")

    acn = n**2 + n + m

    return acn


def acn2nm(acn):
    """
    Calculate the spherical harmonic order n and degree m for a linear
    coefficient index, according to the Ambisonics Channel Convention [1]_.

    .. math::

        n = \\lfloor \\sqrt{acn + 1} \\rfloor - 1

        m = acn - n^2 -n


    References
    ----------
    .. [1]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A Suggested Ambisonics
            Format (revised by F. Zotter),” International Symposium on Ambisonics and Spherical
            Acoustics, vol. 3, pp. 1–11, 2011.


    Parameters
    ----------
    n : integer, ndarray
        Spherical harmonic order
    m : integer, ndarray
        Spherical harmonic degree

    Returns
    -------
    acn : integer, ndarray
        Linear index

    """
    acn = np.asarray(acn, dtype=np.int)

    n = (np.ceil(np.sqrt(acn + 1)) - 1)
    m = acn - n**2 - n

    n = n.astype(np.int, copy=False)
    m = m.astype(np.int, copy=False)

    return n, m


cdef int acn2n(int acn) nogil:
    """ACN to n conversion with c speed and without global interpreter lock.
    """
    cdef int n
    n = <int>ceil(sqrt(<double>acn + 1)) - 1

cdef int acn2m(int acn) nogil:
    """ACN to m conversion with c speed and without global interpreter lock.
    """
    cdef int n = acn2n(acn)
    cdef int m = acn - <int>pow(n, 2) - n
    return m


@cython.boundscheck(False)
@cython.wraparound(False)
def spherical_harmonic_basis(int n_max, coords):
    """
    Calulcates the complex valued spherical harmonic basis matrix of order Nmax
    for a set of points given by their elevation and azimuth angles.
    The spherical harmonic functions are fully normalized (N3D) and include the
    Condon-Shotley phase term (-1)^m [2]_, [3]_.

    .. math::

        Y_n^m(\\theta, \\phi) = \\sqrt{\\frac{2n+1}{4\\pi} \\frac{(n-m)!}{(n+m)!}} P_n^m(\\cos \\theta) e^{i m \\phi}

    References
    ----------
    .. [2]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [3]  B. Rafaely, Fundamentals of Spherical Array Processing, vol. 8. Springer, 2015.


    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    coordinates : Coordinates
        Coordinate object with sampling points for which the basis matrix is
        calculated

    Returns
    -------
    Y : double, ndarray, matrix
        Complex spherical harmonic basis matrix
    """
    cdef cnp.ndarray[double, ndim=1] elevation
    cdef cnp.ndarray[double, ndim=1] azimuth

    if coords.elevation.ndim < 1:
        elevation = coords.elevation[np.newaxis]
        azimuth = coords.azimuth[np.newaxis]
    else:
        elevation = coords.elevation
        azimuth = coords.azimuth

    cdef Py_ssize_t n_points = elevation.shape[0]
    cdef Py_ssize_t n_coeff = (n_max+1)**2
    cdef cnp.ndarray[complex, ndim=2] basis = \
        np.zeros((n_points, n_coeff), dtype=np.complex)
    cdef complex[:, ::1] memview_basis = basis

    cdef double[::1] memview_azi = azimuth
    cdef double[::1] memview_ele = elevation

    cdef Py_ssize_t aa, ii, order, degree
    for aa in range(0, n_points):
        for ii in prange(0, n_coeff, nogil=True):
            order = <int>(ceil(sqrt(<double>ii + 1.0)) - 1)
            degree = ii - order**2 - order

            memview_basis[aa, ii] = spherical_harmonic_function(order, degree, memview_ele[aa], memview_azi[aa])

    return basis


@cython.boundscheck(False)
@cython.wraparound(False)
def spherical_harmonic_basis_real(int n_max, coords):
    """
    Calulcates the real valued spherical harmonic basis matrix of order Nmax
    for a set of points given by their elevation and azimuth angles.
    The spherical harmonic functions are fully normalized (N3D) and follow
    the AmbiX phase convention [1]_.

    .. math::

        Y_n^m(\\theta, \\phi) = \\sqrt{\\frac{2n+1}{4\\pi} \\frac{(n-|m|)!}{(n+|m|)!}} P_n^{|m|}(\\cos \\theta)
        \\begin{cases}
            \displaystyle \\cos(|m|\\phi),  & \\text{if $m \\ge 0$} \\newline
            \displaystyle \\sin(|m|\\phi) ,  & \\text{if $m < 0$}
        \\end{cases}

    References
    ----------
    .. [1]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A Suggested Ambisonics
            Format (revised by F. Zotter),” International Symposium on Ambisonics and Spherical
            Acoustics, vol. 3, pp. 1–11, 2011.


    Parameters
    ----------
    n : integer
        Spherical harmonic order
    coordinates : Coordinates
        Coordinate object with sampling points for which the basis matrix is
        calculated

    Returns
    -------
    Y : double, ndarray, matrix
        Real valued spherical harmonic basis matrix


    """
    if coords.elevation.ndim < 1:
        elevation = coords.elevation[np.newaxis]
        azimuth = coords.azimuth[np.newaxis]
    else:
        elevation = coords.elevation
        azimuth = coords.azimuth

    cdef Py_ssize_t n_points = elevation.shape[0]
    cdef Py_ssize_t n_coeff = (n_max+1)**2
    cdef cnp.ndarray[double, ndim=2] basis = \
        np.zeros((n_points, n_coeff), dtype=np.double)
    cdef double[:, ::1] memview_basis = basis

    cdef double[::1] memview_azi = azimuth
    cdef double[::1] memview_ele = elevation

    cdef Py_ssize_t aa, ii, order, degree
    for aa in range(0, n_points):
        for ii in prange(0, n_coeff, nogil=True):
            order = <int>(ceil(sqrt(<double>ii + 1.0)) - 1)
            degree = ii - order**2 - order

            memview_basis[aa, ii] = spherical_harmonic_function_real(order, degree, memview_ele[aa], memview_azi[aa])

    return basis


def modal_strength(int n_max,
                   cnp.ndarray[double, ndim=1] kr,
                   arraytype='open'):
    """
    Modal strenght function for microphone arrays.

    .. math::

        b(kr) = TODO

    Notes
    -----
    This implementation uses the second order Hankel function, see [4]_ for an
    overview of the corresponding sign conventions.

    References
    ----------
    .. [4]  V. Tourbabin and B. Rafaely, “On the Consistent Use of Space and Time
            Conventions in Array Processing,” vol. 101, pp. 470–473, 2015.


    Parameters
    ----------
    n : integer, ndarray
        Spherical harmonic order
    kr : double, ndarray
        Wave number * radius
    arraytype : string
        Array configuration. Can be a microphones mounted on a rigid sphere,
        on a virtual open sphere or cardioid microphones on an open sphere.

    Returns
    -------
    B : double, ndarray
        Modal strenght diagonal matrix

    """
    arraytypes = {'open': 0, 'rigid': 1, 'cardioid': 2}
    cdef int n_coeff = (n_max+1)**2
    cdef int n_bins = kr.shape[0]
    cdef complex *mat = make_modal_strength(n_max, &kr[0], n_bins, arraytypes.get(arraytype))
    cdef complex[:, :, ::1] mv = <complex[:n_bins, :n_coeff, :n_coeff]>mat
    cdef cnp.ndarray arr = np.asarray(mv)
    set_base(arr, mat)
    return arr

def aperture_spherical_cap(int n_max,
                           double rad_sphere,
                           double rad_cap):
    """
    Aperture function for a vibrating cap in a rigid sphere.

    .. math::

        A(r, \\alpha) = TODO


    References
    ----------
    TODO: Add reference

    Parameters
    ----------
    n : integer, ndarray
        Spherical harmonic order
    r : double, ndarray
        Sphere radius
    alpha : double
        Aperture angle

    Returns
    -------
    A : double, ndarray
        Aperture function diagonal matrix

    """
    cdef double angle_cap = np.arcsin(rad_cap / rad_sphere)
    cdef double arg = np.cos(angle_cap)
    cdef int n_sh = (n_max+1)**2

    cdef double legendre_plus, legendre_minus

    cdef cnp.ndarray[double, ndim=2] aperture = np.zeros((n_sh, n_sh), dtype=np.double)
    aperture[0,0] = (1-arg)*2*np.pi**2

    cdef int n, m
    for n in range(1, n_max+1):
        legendre_minus = legendre_polynomial(n-1, arg)
        legendre_plus = legendre_polynomial(n+1, arg)
        for m in range(-n, n+1):
            acn = nm2acn(n, m)
            aperture[acn, acn] = (legendre_minus - legendre_plus) * 4 * np.pi**2 / (2*n+1)

    return aperture

def radiation_from_sphere(int n_max,
                          double rad_sphere,
                          cnp.ndarray[double, ndim=1] k,
                          double distance):
    cdef int n_sh = (n_max+1)**2

    cdef double rho = 1.2
    cdef double c = 343.0
    cdef complex hankel, hankel_prime, radiation_order
    cdef int n_bins = k.shape[0]
    cdef cnp.ndarray[complex, ndim=3] radiation = np.zeros((n_bins, n_sh, n_sh), dtype=np.complex)

    cdef int n, m, kk
    for kk in range(0, n_bins):
        for n in range(0, n_max+1):
            hankel = sph_hankel_2(n, k[kk]*distance)
            hankel_prime = sph_hankel_2_prime(n, k[kk]*rad_sphere)
            radiation_order = hankel/hankel_prime * 1j * rho * c
            for m in range(-n, n+1):
                acn = nm2acn(n, m)
                radiation[kk, acn, acn] = radiation_order

    return radiation

