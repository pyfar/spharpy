import numpy as np
import scipy.special as special


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


def spherical_harmonic_basis(n_max, coords):
    """
    Calulcates the complex valued spherical harmonic basis matrix of order Nmax
    for a set of points given by their elevation and azimuth angles.
    The spherical harmonic functions are fully normalized (N3D) and include the
    Condon-Shotley phase term :math:`(-1)^m` [2]_, [3]_.

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

    n_coeff = (n_max+1)**2

    basis = np.zeros((coords.n_points, n_coeff), dtype=np.complex)

    for acn in range(0, n_coeff):
        order, degree = acn2nm(acn)
        basis[:, acn] = special.sph_harm(
            degree,
            order,
            coords.azimuth,
            coords.elevation)

    return basis


def spherical_harmonic_basis_real(n_max, coords):
    r"""
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
    n_coeff = (n_max+1)**2

    basis = np.zeros((coords.n_points, n_coeff), dtype=np.double)

    for acn in range(0, n_coeff):
        order, degree = acn2nm(acn)
        basis[:, acn] = sph_harm_real(
            order,
            degree,
            coords.elevation,
            coords.azimuth)

    return basis


def sph_harm_real(n, m, theta, phi):
    """Real valued spherical harmonic function.
    TODO: Move this to a new special function module

    Parameters
    ----------
    n : TODO
    m : TODO
    theta : TODO
    phi : TODO

    Returns
    -------
    TODO

    """

    # careful here, scipy uses phi as the elevation angle and
    # theta as the azimuth angle
    Y_nm_cplx = special.sph_harm(m, n, phi, theta)

    if m == 0:
        Y_nm = np.real(Y_nm_cplx)
    elif m > 0:
        Y_nm = np.real(Y_nm_cplx) * np.sqrt(2)
    elif m < 0:
        Y_nm = np.imag(Y_nm_cplx) * np.sqrt(2) * np.float(-1)**(m+1)

    return Y_nm * (np.float(-1)**(m))


def modal_strength(n_max,
                   kr,
                   arraytype='rigid'):
    r"""
    Modal strenght function for microphone arrays.

    .. math::

        b(kr) =
        \\begin{cases}
            \displaystyle 4\\pi i^n j_n(kr),  & \\text{open} \\newline
            \displaystyle  4\\pi i^{(n-1)} \\frac{1}{(kr)^2 h_n^\\prime(kr)},  & \\text{rigid} \\newline
            \displaystyle  4\\pi i^n (j_n(kr) - i j_n^\\prime(kr)),  & \\text{cardioid}
        \\end{cases}


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
        Modal strength diagonal matrix

    """
    arraytypes = {'open': 0, 'rigid': 1, 'cardioid': 2}
    config = arraytypes.get(arraytype)
    n_coeff = (n_max+1)**2
    n_bins = kr.shape[0]

    modal_strength_mat = np.zeros((n_bins, n_coeff, n_coeff), dtype=np.complex)

    for k in range(0, n_bins):
        for n in range(0, n_max+1):
            bn = _modal_strength(n, kr[k], config)
            for m in range(-n, n+1):
                acn = n*n + n + m
                modal_strength_mat[k, acn, acn] = bn

    return np.squeeze(modal_strength_mat)


def spherical_hn(n, z, kind=2):
    if kind == 1:
        hankel = special.hankel1(n+0.5, z)
    elif kind == 2:
        hankel = special.hankel2(n+0.5, z)
    return np.sqrt(np.pi/2/z) * hankel


def _modal_strength(n, kr, config):
    """Helper function for the calculation of the modal strength for
    plane waves"""
    if config == 0:
        ms = 4*np.pi*pow(1.0j, n) * special.spherical_jn(n, kr)
    # elif config == 1:
    #     modal_strength = 4*np.pi*pow(1.0j, n-1) / \
    #             _special.sph_hankel_2_prime(n, kr) / (kr)**2
    # elif config == 2:
    #     modal_strength = 4*np.pi*pow(1.0j, n) * \
    #             (special.spherical_jn(n, kr) - 1.0j * _special.sph_bessel_prime(n, kr))
    else:
        raise ValueError("Invalid configuration.")

    return ms
