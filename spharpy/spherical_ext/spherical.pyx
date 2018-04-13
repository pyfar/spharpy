# distutils: language = c++
# cython: embedsignature = True
"""
Spherical extension module docstring
"""


import numpy as np
cimport numpy as cnp
from libc.stdlib cimport free
import cython

cdef extern from "spherical_harmonics.h":
    int pyramid2linear(int order, int degree);
    int linear2pyramid_order(int linear_index);
    int linear2pyramid_degree(int linear_index, unsigned order);
    double complex spherical_harmonic_function_cpp(unsigned order, int degree, double theta, double phi);
    complex* make_spherical_harmonics_basis(unsigned Nmax, double *theta, double *phi, unsigned npoints);
    double* make_spherical_harmonics_basis_real(unsigned n_max, double *theta, double *phi, unsigned n_points);
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


def spherical_harmonic_function(int n, int m, double theta, double phi):
    return spherical_harmonic_function_cpp(n, m, theta, phi)

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
    n = np.asarray(n, dtype=np.int32)
    m = np.asarray(m, dtype=np.int32)
    n_acn = m.size

    if not (n.size == m.size):
        return -1
    if not n.shape:
        n = n[np.newaxis]
    if not m.shape:
        m = m[np.newaxis]

    cdef cnp.ndarray[int, ndim=1] acn = np.zeros(n_acn, dtype=np.int32)

    cdef int idx
    for idx in range(0, n_acn):
        acn[idx] = pyramid2linear(n[idx], m[idx])

    return np.squeeze(acn)


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

    acn = np.asarray(acn, dtype=np.int32)
    n_acn = acn.size
    if not acn.shape:
        acn = acn[np.newaxis]

    cdef cnp.ndarray[int, ndim=1] n = np.zeros(n_acn, dtype=np.int32)
    cdef cnp.ndarray[int, ndim=1] m = np.zeros(n_acn, dtype=np.int32)

    cdef int idx
    for idx in range(0, n_acn):
        n[idx] = linear2pyramid_order(acn[idx])
        m[idx] = linear2pyramid_degree(acn[idx], n[idx])

    return np.squeeze(n), np.squeeze(m)


@cython.boundscheck(False)
@cython.wraparound(False)
def spherical_harmonic_basis(unsigned Nmax,
                             cnp.ndarray[double, ndim=1] theta,
                             cnp.ndarray[double, ndim=1] phi):
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
    n : integer
        Spherical harmonic order
    theta : double ndarray
        Elevation angle [0, pi]
    phi : double ndarray
        Azimuth angle [0, 2pi]

    Returns
    -------
    Y : double, ndarray, matrix
        Complex spherical harmonic basis matrix
    """

    cdef unsigned n_points = theta.shape[0]
    cdef int n_coeff = (Nmax+1)*(Nmax+1)
    cdef complex *mat = make_spherical_harmonics_basis(Nmax, &theta[0], &phi[0], n_points)
    cdef complex[:, ::1] mv = <complex[:n_points, :n_coeff]>mat
    cdef cnp.ndarray arr = np.asarray(mv)
    set_base(arr, mat)
    return arr

def spherical_harmonic_basis_real(unsigned Nmax,
                                  cnp.ndarray[double, ndim=1] theta,
                                  cnp.ndarray[double, ndim=1] phi):
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
    theta : double ndarray
        Elevation angle [0, pi]
    phi : double ndarray
        Azimuth angle [0, 2pi]

    Returns
    -------
    Y : double, ndarray, matrix
        Real valued spherical harmonic basis matrix


    """
    cdef unsigned n_points = theta.shape[0]
    cdef int n_coeff = (Nmax+1)*(Nmax+1)
    cdef double *mat = make_spherical_harmonics_basis_real(Nmax, &theta[0], &phi[0], n_points)
    cdef double[:, ::1] mv = <double[:n_points, :n_coeff]>mat
    cdef cnp.ndarray arr = np.asarray(mv)
    set_base(arr, mat)
    return arr

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

