"""
Subpackage implementing or wrapping special functions required in the
spharpy package.
"""

from itertools import count
import numpy as np
import scipy.special as _spspecial
from scipy.optimize import brentq


def spherical_bessel(n, z, derivative=False):
    r"""
    Spherical bessel function of order n evaluated at z.

    .. math::

        j_n(z) = \sqrt{\frac{\pi}{2z}} J_{n+\frac{1}{2}} (z)

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

    if z.ndim <= 1 or n.ndim <= 1:
        bessel = np.squeeze(bessel)

    return bessel


def spherical_bessel_zeros(n_max, n_zeros):
    """Compute the zeros of the spherical Bessel function.
    This function will always start at order zero which is equal
    to sin(x)/x and iteratively compute the roots for higher orders.
    The roots are computed using Brents algorithm from the scipy package.

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
        return _spspecial.spherical_jn(n, x)

    zerosj = np.zeros((n_max+1, n_zeros), dtype=np.double)
    zerosj[0] = np.arange(1, n_zeros+1)*np.pi
    points = np.arange(1, n_zeros+n_max+1)*np.pi

    roots = np.zeros(n_zeros+n_max, dtype=np.double)
    for i in range(1, n_max+1):
        for j in range(n_zeros+n_max-i):
            roots[j] = brentq(func, points[j], points[j+1], (i,), maxiter=5000)
        points = roots
        zerosj[i, :n_zeros] = roots[:n_zeros]

    return zerosj


def spherical_hankel(n, z, kind=2, derivative=False):
    r"""
    Spherical Hankel function of order n evaluated at z.

    .. math::

        j_n(z) = \sqrt{\frac{\pi}{2z}} J_{n+\frac{1}{2}} (z)

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

    if z.ndim <= 1 or n.ndim <= 1:
        hankel = np.squeeze(hankel)

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


def spherical_harmonic(n, m, theta, phi):
    """The spherical harmonics of order n and degree m.

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
    spherical_harmonic : ndarray, double
        The complex valued spherial harmonic of order n and degree m

    Note
    ----
    This function wraps the spherical harmonic implementation from scipy.
    The only difference is that we return zeros instead of nan values
    if $n < |m|$.

    """
    theta = np.asarray(theta, dtype=np.double)
    phi = np.asarray(phi, dtype=np.double)

    if n < np.abs(m):
        sph_harm = np.zeros(theta.shape)
    else:
        sph_harm = _spspecial.sph_harm(m, n, phi, theta)
    return sph_harm


def spherical_harmonic_real(n, m, theta, phi):
    r"""Real valued spherical harmonic function of order n and degree m
    evaluated at the angles theta and phi.
    The spherical harmonic functions are fully normalized (N3D) and follow
    the AmbiX phase convention [1]_.

    .. math::

        Y_n^m(\theta, \phi) = \sqrt{\frac{2n+1}{4\pi}
        \frac{(n-|m|)!}{(n+|m|)!}} P_n^{|m|}(\cos \theta)
        \begin{cases}
            \displaystyle \cos(|m|\phi),  & \text{if $m \ge 0$} \newline
            \displaystyle \sin(|m|\phi) ,  & \text{if $m < 0$}
        \end{cases}

    References
    ----------
    .. [1]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
            Suggested Ambisonics Format (revised by F. Zotter),” International
            Symposium on Ambisonics and Spherical Acoustics,
            vol. 3, pp. 1–11, 2011.



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
    spherical_harmonic : ndarray, double
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


def spherical_harmonic_derivative_phi(n, m, theta, phi):
    """Calculate the derivative of the spherical harmonics with respect to
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
    if m == 0 or n == 0:
        res = np.zeros(phi.shape, dtype=np.complex)
    else:
        res = spherical_harmonic(n, m, theta, phi) * 1j * m

    return res


def spherical_harmonic_gradient_phi(n, m, theta, phi):
    """Calculate the derivative of the spherical harmonics with respect to
    the azimuth angle phi divided by sin(theta)

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
        res = np.zeros(theta.shape, dtype=np.complex)
    else:
        factor = np.sqrt((2*n+1)/(2*n-1))/2
        exp_phi = np.exp(1j*phi)
        first = np.sqrt((n+m)*(n+m-1)) * exp_phi * \
            spherical_harmonic(n-1, m-1, theta, phi)
        second = np.sqrt((n-m) * (n-m-1)) / exp_phi * \
            spherical_harmonic(n-1, m+1, theta, phi)
        Ynm_sin_theta = (-1) * factor * (first + second)
        res = Ynm_sin_theta * 1j

    return res


def spherical_harmonic_derivative_theta(n, m, theta, phi):
    """Calculate the derivative of the spherical harmonics with respect to
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

    """
    if n == 0:
        res = np.zeros(theta.shape, dtype=np.complex)
    else:
        exp_phi = np.exp(1j*phi)
        first = np.sqrt((n-m+1) * (n+m)) * exp_phi * \
            spherical_harmonic(n, m-1, theta, phi)
        second = np.sqrt((n-m) * (n+m+1)) / exp_phi * \
            spherical_harmonic(n, m+1, theta, phi)
        res = (first-second)/2 * (-1)

    return res


def legendre_function(n, m, z, cs_phase=True):
    r"""Legendre function of order n and degree m with argument z.

    .. math::

        P_n^m(z) = (-1)^m(1-z^2)^{m/2}\frac{d^m}{dz^m}P_n{z}

    where the Condon-Shotley phase term $(-1)^m$ is dropped when cs_phase=False
    is used.

    Parameters
    ----------
    n : int
        The order
    m : int
        The degree
    z : ndarray, double
        The argument as an array
    cs_phase : bool, optional
        Whether to use include the Condon-Shotley phase term (-1)^m or not

    Returns
    -------
    legendre : ndarray, double
        The Legendre function. This will return zeros if $|m| > n$.

    Note
    ----
    This is a wrapper for the Legendre function implementation from scipy. The
    scipy implementation uses the Condon-Shotley phase. Therefore, the sign
    needs to be flipped here for uneven degrees when dropping the
    Condon-Shotley phase.

    """
    z = np.atleast_1d(z)

    if np.abs(m) > n:
        legendre = np.zeros(z.shape)
    else:
        legendre = np.zeros(z.shape)
        for idx, arg in zip(count(), z):
            leg, _ = _spspecial.lpmn(m, n, arg)
            if np.mod(m, 2) != 0 and not cs_phase:
                legendre[idx] = -leg[-1, -1]
            else:
                legendre[idx] = leg[-1, -1]
    return legendre


def spherical_harmonic_normalization(n, m, norm='full'):
    r"""The normalization factor for real valued spherical harmonics.

    .. math::

        N_n^m = \sqrt{\frac{2n+1}{4\pi}\frac{(n-m)!}{(n+m)!}}

    Parameters
    ----------
    n : int
        The spherical harmonic order.
    m : int
        The spherical harmonic degree.
    norm : 'full', 'semi', optional
        Normalization to use. Can be either fully normalzied on the sphere or
        semi-normalized.

    Returns
    -------
    norm : double
        The normalization factor.


    """
    if np.abs(m) > n:
        factor = 0.0
    else:
        if norm == 'full':
            z = n+m+1
            factor = _spspecial.poch(z, -2*m)
            factor *= (2*n+1)/(4*np.pi)
            if int(m) != 0:
                factor *= 2
            factor = np.sqrt(factor)
        elif norm == 'semi':
            z = n+m+1
            factor = _spspecial.poch(z, -2*m)
            if int(m) != 0:
                factor *= 2
            factor = np.sqrt(factor)
        else:
            raise ValueError("Unknown normalization.")
    return factor


def spherical_harmonic_derivative_theta_real(n, m, theta, phi):
    r"""The derivative of the real valued spherical harmonics with respect
    to the elevation angle $\theta$.

    Parameters
    ----------
    n : int
        The spherical harmonic order.
    m : int
        The spherical harmonic degree.
    theta : ndarray, double
        The elevation angle
    phi : ndarray, double
        The azimuth angle


    Returns
    -------
    derivative : ndarray, double
        The derivative

    Note
    ----
    This implementation does not include the Condon-Shotley phase term.

    """

    m_abs = np.abs(m)
    if n == 0:
        res = np.zeros(theta.shape, dtype=np.double)
    else:
        first = (n+m_abs)*(n-m_abs+1) * \
            legendre_function(
                n,
                m_abs-1,
                np.cos(theta),
                cs_phase=False)
        second = legendre_function(
            n,
            m_abs+1,
            np.cos(theta),
            cs_phase=False)
        legendre_diff = 0.5*(first - second)

        N_nm = spherical_harmonic_normalization(n, m_abs)

        if m < 0:
            phi_term = np.sin(m_abs*phi)
        else:
            phi_term = np.cos(m_abs*phi)

        res = N_nm * legendre_diff * phi_term

    return res


def spherical_harmonic_derivative_phi_real(n, m, theta, phi):
    r"""The derivative of the real valued spherical harmonics with respect
    to the azimuth angle $\phi$.

    Parameters
    ----------
    n : int
        The spherical harmonic order.
    m : int
        The spherical harmonic degree.
    theta : ndarray, double
        The elevation angle
    phi : ndarray, double
        The azimuth angle


    Returns
    -------
    derivative : ndarray, double
        The derivative

    Note
    ----
    This implementation does not include the Condon-Shotley phase term.

    """
    m_abs = np.abs(m)
    if m == 0:
        res = np.zeros(theta.shape, dtype=np.double)
    else:
        legendre = legendre_function(n, m_abs, np.cos(theta), cs_phase=False)
        N_nm = spherical_harmonic_normalization(n, m_abs)

        if m < 0:
            phi_term = np.cos(m_abs*phi) * m_abs
        else:
            phi_term = -np.sin(m_abs*phi) * m_abs

        res = N_nm * legendre * phi_term

    return res


def spherical_harmonic_gradient_phi_real(n, m, theta, phi):
    r"""The gradient of the real valued spherical harmonics with respect
    to the azimuth angle $\phi$.

    Parameters
    ----------
    n : int
        The spherical harmonic order.
    m : int
        The spherical harmonic degree.
    theta : ndarray, double
        The elevation angle
    phi : ndarray, double
        The azimuth angle


    Returns
    -------
    derivative : ndarray, double
        The derivative

    Note
    ----
    This implementation does not include the Condon-Shotley phase term.

    References
    ----------
    .. [1]  J. Du, C. Chen, V. Lesur, and L. Wang, “Non-singular spherical
            harmonic expressions of geomagnetic vector and gradient tensor
            fields in the local north-oriented reference frame,” Geoscientific
            Model Development, vol. 8, no. 7, pp. 1979–1990, Jul. 2015.

    """
    m_abs = np.abs(m)
    if m == 0:
        res = np.zeros(theta.shape, dtype=np.double)
    else:
        first = (n+m_abs)*(n+m_abs-1) * \
            legendre_function(
                n-1,
                m_abs-1,
                np.cos(theta),
                cs_phase=False)
        second = legendre_function(
            n-1,
            m_abs+1,
            np.cos(theta),
            cs_phase=False)
        legendre_diff = 0.5*(first + second)
        N_nm = spherical_harmonic_normalization(n, m_abs)

        if m < 0:
            phi_term = np.cos(m_abs*phi)
        else:
            phi_term = -np.sin(m_abs*phi)

        res = N_nm * legendre_diff * phi_term

    return res


def legendre_coefficients(order):
    """Calculate the coefficients of a Legendre polynomial of the
    specified order.

    Parameters
    ----------
    order : int
        The order of the polynomial

    Returns
    -------
    coefficients : ndarray, double
        The coefficients of the polynomial

    """
    leg = np.polynomial.legendre.Legendre.basis(order)
    coefficients = np.polynomial.legendre.leg2poly(leg.coef)

    return coefficients


def chebyshev_coefficients(order):
    """Calculate the coefficients of a Chebyshev polynomial of the
    specified order.

    Parameters
    ----------
    order : int
        The order of the polynomial

    Returns
    -------
    coefficients : ndarray, double
        The coefficients of the polynomial

    """
    cheb = np.polynomial.chebyshev.Chebyshev.basis(order)
    coefficients = np.polynomial.chebyshev.cheb2poly(cheb.coef)

    return coefficients
