import numpy as np
import scipy.special as special
import spharpy.special as _special


def acn2nm(acn):
    r"""
    Calculate the spherical harmonic order n and degree m for a linear
    coefficient index, according to the Ambisonics Channel Convention [1]_.

    .. math::

        n = \lfloor \sqrt{acn + 1} \rfloor - 1

        m = acn - n^2 -n


    References
    ----------
    .. [1]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
            Suggested Ambisonics Format (revised by F. Zotter),” International
            Symposium on Ambisonics and Spherical Acoustics,
            vol. 3, pp. 1–11, 2011.


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
    .. [1]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
            Suggested Ambisonics Format (revised by F. Zotter),” International
            Symposium on Ambisonics and Spherical Acoustics,
            vol. 3, pp. 1–11, 2011.


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

    if not (n.size == m.size):
        raise ValueError("n and m need to be of the same size")

    acn = n**2 + n + m

    return acn


def spherical_harmonic_basis(n_max, coords):
    r"""
    Calulcates the complex valued spherical harmonic basis matrix of order Nmax
    for a set of points given by their elevation and azimuth angles.
    The spherical harmonic functions are fully normalized (N3D) and include the
    Condon-Shotley phase term :math:`(-1)^m` [2]_, [3]_.

    .. math::

        Y_n^m(\theta, \phi) = \sqrt{\frac{2n+1}{4\pi}
        \frac{(n-m)!}{(n+m)!}} P_n^m(\cos \theta) e^{i m \phi}

    References
    ----------
    .. [2]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [3]  B. Rafaely, Fundamentals of Spherical Array Processing, vol. 8.
            Springer, 2015.


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
        basis[:, acn] = _special.spherical_harmonic(
            order,
            degree,
            coords.elevation,
            coords.azimuth)

    return basis


def spherical_harmonic_basis_gradient(n_max, coords):
    r"""
    Calulcates the gradient on the unit sphere of the complex valued spherical
    harmonic basis matrix of order N for a set of points given by their
    elevation and azimuth angles.
    The spherical harmonic functions are fully normalized (N3D) and include the
    Condon-Shotley phase term :math:`(-1)^m` [2]_. This implementation avoids
    singularities at the poles using identities derived in [5]_.


    References
    ----------
    .. [2]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [9]  J. Du, C. Chen, V. Lesur, and L. Wang, “Non-singular spherical
            harmonic expressions of geomagnetic vector and gradient tensor
            fields in the local north-oriented reference frame,” Geoscientific
            Model Development, vol. 8, no. 7, pp. 1979–1990, Jul. 2015.


    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    coordinates : Coordinates
        Coordinate object with sampling points for which the basis matrix is
        calculated

    Returns
    -------
    grad_elevation : double, ndarray, matrix
        Gradient with regard to the elevation angle.
    grad_azimuth : double, ndarray, matrix
        Gradient with regard to the azimuth angle.


    """
    n_points = coords.n_points
    n_coeff = (n_max+1)**2
    theta = coords.elevation
    phi = coords.azimuth
    grad_theta = np.zeros((n_points, n_coeff), dtype=np.complex)
    grad_phi = np.zeros((n_points, n_coeff), dtype=np.complex)

    for acn in range(0, n_coeff):
        n, m = acn2nm(acn)

        grad_theta[:, acn] = \
            _special.spherical_harmonic_derivative_theta(
                n, m, theta, phi)
        grad_phi[:, acn] = \
            _special.spherical_harmonic_gradient_phi(
                n, m, theta, phi)

    return grad_theta, grad_phi


def spherical_harmonic_basis_real(n_max, coords):
    r"""
    Calulcates the real valued spherical harmonic basis matrix of order Nmax
    for a set of points given by their elevation and azimuth angles.
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
        basis[:, acn] = _special.spherical_harmonic_real(
            order,
            degree,
            coords.elevation,
            coords.azimuth)

    return basis


def spherical_harmonic_basis_gradient_real(n_max, coords):
    r"""
    Calulcates the gradient on the unit sphere of the real valued spherical
    harmonic basis matrix of order N for a set of points given by their
    elevation and azimuth angles.
    The spherical harmonic functions are fully normalized (N3D) and follow
    the AmbiX phase convention [1]_. This implementation avoids
    singularities at the poles using identities derived in [5]_.


    References
    ----------
    .. [1]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
            Suggested Ambisonics Format (revised by F. Zotter),” International
            Symposium on Ambisonics and Spherical Acoustics,
            vol. 3, pp. 1–11, 2011.
    .. [9]  J. Du, C. Chen, V. Lesur, and L. Wang, “Non-singular spherical
            harmonic expressions of geomagnetic vector and gradient tensor
            fields in the local north-oriented reference frame,” Geoscientific
            Model Development, vol. 8, no. 7, pp. 1979–1990, Jul. 2015.

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
    n_points = coords.n_points
    n_coeff = (n_max+1)**2
    theta = coords.elevation
    phi = coords.azimuth
    grad_theta = np.zeros((n_points, n_coeff), dtype=np.double)
    grad_phi = np.zeros((n_points, n_coeff), dtype=np.double)

    for acn in range(0, n_coeff):
        n, m = acn2nm(acn)

        grad_theta[:, acn] = \
            _special.spherical_harmonic_derivative_theta_real(
                n, m, theta, phi)
        grad_phi[:, acn] = \
            _special.spherical_harmonic_gradient_phi_real(
                n, m, theta, phi)

    return grad_theta, grad_phi


def modal_strength(n_max,
                   kr,
                   arraytype='rigid'):
    r"""
    Modal strenght function for microphone arrays.

    .. math::

        b(kr) =
        \begin{cases}
            \displaystyle 4\pi i^n j_n(kr),  & \text{open} \newline
            \displaystyle  4\pi i^{(n-1)} \frac{1}{(kr)^2 h_n^\prime(kr)},
                & \text{rigid} \newline
            \displaystyle  4\pi i^n (j_n(kr) - i j_n^\prime(kr)),
                & \text{cardioid}
        \end{cases}


    Notes
    -----
    This implementation uses the second order Hankel function, see [4]_ for an
    overview of the corresponding sign conventions.

    References
    ----------
    .. [4]  V. Tourbabin and B. Rafaely, “On the Consistent Use of Space and
            Time Conventions in Array Processing,” vol. 101, pp. 470–473, 2015.


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
    n_coeff = (n_max+1)**2
    n_bins = kr.shape[0]

    modal_strength_mat = np.zeros((n_bins, n_coeff, n_coeff), dtype=np.complex)

    for n in range(0, n_max+1):
        bn = _modal_strength(n, kr, arraytype)
        for m in range(-n, n+1):
            acn = n*n + n + m
            modal_strength_mat[:, acn, acn] = bn

    return np.squeeze(modal_strength_mat)


def _modal_strength(n, kr, config):
    """Helper function for the calculation of the modal strength for
    plane waves"""
    if config == 'open':
        ms = 4*np.pi*pow(1.0j, n) * _special.spherical_bessel(n, kr)
    elif config == 'rigid':
        ms = 4*np.pi*pow(1.0j, n+1) / \
            _special.spherical_hankel(n, kr, derivative=True) / (kr)**2
    elif config == 'cardioid':
        ms = 4*np.pi*pow(1.0j, n) * \
            (_special.spherical_bessel(n, kr) -
                1.0j * _special.spherical_bessel(n, kr, derivative=True))
    else:
        raise ValueError("Invalid configuration.")

    return ms


def aperture_vibrating_spherical_cap(
        n_max,
        rad_sphere,
        rad_cap):
    r"""
    Aperture function for a vibrating cap with radius :math:`r_c` in a rigid
    sphere with radius :math:`r_s` [5]_, [6]_

    .. math::

        a_n (r_{s}, \alpha) =
        \begin{cases}
            \displaystyle \cos\left(\alpha\right)
                P_n\left[ \cos\left(\alpha\right) \right] -
                P_{n-1}\left[ \cos\left(\alpha\right) \right],
                & {n>0} \newline
            \displaystyle  1 - \cos(\alpha),  & {n=0}
        \end{cases}

    where :math:`\alpha = \arcsin \left(\frac{r_c}{r_s} \right)` is the
    aperture angle.


    References
    ----------
    .. [5]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [6]  F. Zotter, A. Sontacchi, and R. Höldrich, “Modeling a spherical
            loudspeaker system as multipole source,” in Proceedings of the 33rd
            DAGA German Annual Conference on Acoustics, 2007, pp. 221–222.


    Parameters
    ----------
    n_max : integer, ndarray
        Maximal spherical harmonic order
    r_sphere : double, ndarray
        Radius of the sphere
    r_cap : double
        Radius of the vibrating cap

    Returns
    -------
    A : double, ndarray
        Aperture function in diagonal matrix form with shape
        :math:`[(n_{max}+1)^2~\times~(n_{max}+1)^2]`

    """
    angle_cap = np.arcsin(rad_cap / rad_sphere)
    arg = np.cos(angle_cap)
    n_sh = (n_max+1)**2

    aperture = np.zeros((n_sh, n_sh), dtype=np.double)

    aperture[0, 0] = (1-arg)*2*np.pi**2
    for n in range(1, n_max+1):
        legendre_minus = special.legendre(n-1)(arg)
        legendre_plus = special.legendre(n+1)(arg)
        for m in range(-n, n+1):
            acn = nm2acn(n, m)
            aperture[acn, acn] = (legendre_minus - legendre_plus) * \
                4 * np.pi**2 / (2*n+1)

    return aperture


def radiation_from_sphere(
        n_max,
        rad_sphere,
        k,
        distance,
        density_medium=1.2,
        speed_of_sound=343.0):
    r"""
    Radiation function in SH for a vibrating sphere including the radiation
    impedance and the propagation to a arbitrary distance from the sphere.
    The sign and phase conventions result in a positive pressure response for
    a positive cap velocity with the intensity vector pointing away from the
    source.

    TODO: This function does not have a test yet.

    References
    ----------
    .. [7]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [8]  F. Zotter, A. Sontacchi, and R. Höldrich, “Modeling a spherical
            loudspeaker system as multipole source,” in Proceedings of the 33rd
            DAGA German Annual Conference on Acoustics, 2007, pp. 221-222.


    Parameters
    ----------
    n_max : integer, ndarray
        Maximal spherical harmonic order
    r_sphere : double, ndarray
        Radius of the sphere
    k : double, ndarray
        Wave number
    distance : double
        Distance from the origin
    density_medium : double
        Density of the medium surrounding the sphere. Default is 1.2 for air.
    speed_of_sound : double
        Speed of sound in m/s

    Returns
    -------
    R : double, ndarray
        Radiation function in diagonal matrix form with shape
        :math:`[K \times (n_{max}+1)^2~\times~(n_{max}+1)^2]`

    """
    n_sh = (n_max+1)**2

    k = np.atleast_1d(k)
    n_bins = k.shape[0]
    radiation = np.zeros((n_bins, n_sh, n_sh), dtype=np.complex)

    for n in range(0, n_max+1):
        hankel = _special.spherical_hankel(n, k*distance, kind=2)
        hankel_prime = _special.spherical_hankel(
            n, k*rad_sphere, kind=2, derivative=True)
        radiation_order = -1j * hankel/hankel_prime * \
            density_medium * speed_of_sound
        for m in range(-n, n+1):
            acn = nm2acn(n, m)
            radiation[:, acn, acn] = radiation_order

    return radiation
