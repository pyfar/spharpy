import numpy as np
import scipy.special as special
import spharpy.special as _special
from spharpy._deprecation import convert_coordinates_to_pyfar


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
    acn = np.asarray(acn, dtype=int)

    n = (np.ceil(np.sqrt(acn + 1)) - 1)
    m = acn - n**2 - n

    n = n.astype(int, copy=False)
    m = m.astype(int, copy=False)

    return n, m


def nm2acn(n, m):
    """
    Calculate the linear index coefficient for a spherical harmonic order n
    and degree m, according to the Ambisonics Channel Convention [2]_.

    .. math::

        acn = n^2 + n + m

    References
    ----------
    .. [2]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
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
    n = np.asarray(n, dtype=int)
    m = np.asarray(m, dtype=int)

    if n.size != m.size:
        raise ValueError("n and m need to be of the same size")

    return n**2 + n + m


def spherical_harmonic_basis(n_max, coords):
    r"""
    Calulcates the complex valued spherical harmonic basis matrix of order Nmax
    for a set of points given by their elevation and azimuth angles.
    The spherical harmonic functions are fully normalized (N3D) and include the
    Condon-Shotley phase term :math:`(-1)^m` [3]_, [4]_.

    .. math::

        Y_n^m(\theta, \\phi) = \\sqrt{\frac{2n+1}{4\pi}
        \frac{(n-m)!}{(n+m)!}} P_n^m(\cos \theta) e^{i m \phi}

    References
    ----------
    .. [3]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [4]  B. Rafaely, Fundamentals of Spherical Array Processing, vol. 8.
            Springer, 2015.


    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    coordinates : :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        Coordinate object with sampling points for which the basis matrix is
        calculated

    Returns
    -------
    Y : double, ndarray, matrix
        Complex spherical harmonic basis matrix

    """  # noqa: E501
    coords = convert_coordinates_to_pyfar(coords)

    n_coeff = (n_max+1)**2

    basis = np.zeros((coords.csize, n_coeff), dtype=complex)

    for acn in range(n_coeff):
        order, degree = acn2nm(acn)
        basis[:, acn] = _special.spherical_harmonic(
            order,
            degree,
            coords.colatitude,
            coords.azimuth)

    return basis


def spherical_harmonic_basis_gradient(n_max, coords):
    r"""
    Calulcates the gradient on the unit sphere of the complex valued spherical
    harmonic basis matrix of order N for a set of points given by their
    elevation and azimuth angles.
    The spherical harmonic functions are fully normalized (N3D) and include the
    Condon-Shotley phase term :math:`(-1)^m` [5]_. This implementation avoids
    singularities at the poles using identities derived in [6]_.


    References
    ----------
    .. [5]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [6]  J. Du, C. Chen, V. Lesur, and L. Wang, “Non-singular spherical
            harmonic expressions of geomagnetic vector and gradient tensor
            fields in the local north-oriented reference frame,” Geoscientific
            Model Development, vol. 8, no. 7, pp. 1979–1990, Jul. 2015.


    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    coordinates : :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        Coordinate object with sampling points for which the basis matrix is
        calculated

    Returns
    -------
    grad_elevation : double, ndarray, matrix
        Gradient with regard to the elevation angle.
    grad_azimuth : double, ndarray, matrix
        Gradient with regard to the azimuth angle.


    """ # noqa: 501
    coords = convert_coordinates_to_pyfar(coords)

    n_points = coords.csize
    n_coeff = (n_max+1)**2
    theta = coords.colatitude
    phi = coords.azimuth
    grad_theta = np.zeros((n_points, n_coeff), dtype=complex)
    grad_phi = np.zeros((n_points, n_coeff), dtype=complex)

    for acn in range(n_coeff):
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
    the AmbiX phase convention [7]_.

    .. math::

        Y_n^m(\theta, \phi) = \sqrt{\frac{2n+1}{4\pi}
        \frac{(n-|m|)!}{(n+|m|)!}} P_n^{|m|}(\cos \theta)
        \begin{cases}
            \displaystyle \cos(|m|\phi),  & \text{if $m \ge 0$} \newline
            \displaystyle \sin(|m|\phi) ,  & \text{if $m < 0$}
        \end{cases}

    References
    ----------
    .. [7]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
            Suggested Ambisonics Format (revised by F. Zotter),” International
            Symposium on Ambisonics and Spherical Acoustics,
            vol. 3, pp. 1–11, 2011.


    Parameters
    ----------
    n : integer
        Spherical harmonic order
    coordinates : :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        Coordinate object with sampling points for which the basis matrix is
        calculated

    Returns
    -------
    Y : double, ndarray, matrix
        Real valued spherical harmonic basis matrix


    """ # noqa: 501
    coords = convert_coordinates_to_pyfar(coords)

    n_coeff = (n_max+1)**2

    basis = np.zeros((coords.csize, n_coeff), dtype=float)

    for acn in range(n_coeff):
        order, degree = acn2nm(acn)
        basis[:, acn] = _special.spherical_harmonic_real(
            order,
            degree,
            coords.colatitude,
            coords.azimuth)

    return basis


def spherical_harmonic_basis_gradient_real(n_max, coords):
    r"""
    Calulcates the gradient on the unit sphere of the real valued spherical
    harmonic basis matrix of order N for a set of points given by their
    elevation and azimuth angles.
    The spherical harmonic functions are fully normalized (N3D) and follow
    the AmbiX phase convention [8]_. This implementation avoids
    singularities at the poles using identities derived in [9]_.


    References
    ----------
    .. [8]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
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
    coordinates : :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        Coordinate object with sampling points for which the basis matrix is
        calculated

    Returns
    -------
    Y : double, ndarray, matrix
        Complex spherical harmonic basis matrix

    """ # noqa: 501
    coords = convert_coordinates_to_pyfar(coords)
    n_points = coords.csize
    n_coeff = (n_max+1)**2
    theta = coords.colatitude
    phi = coords.azimuth
    grad_theta = np.zeros((n_points, n_coeff), dtype=float)
    grad_phi = np.zeros((n_points, n_coeff), dtype=float)

    for acn in range(n_coeff):
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
    This implementation uses the second order Hankel function, see [10]_ for an
    overview of the corresponding sign conventions.

    References
    ----------
    .. [10] V. Tourbabin and B. Rafaely, “On the Consistent Use of Space and
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

    modal_strength_mat = np.zeros((n_bins, n_coeff, n_coeff), dtype=complex)

    for n in range(n_max+1):
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
    sphere with radius :math:`r_s` [11]_, [12]_

    .. math::

        a_n (r_{s}, \alpha) = 4 \pi
        \begin{cases}
            \displaystyle \left(2n+1\right)\left[
                P_{n-1} \left(\cos\alpha\right) -
                P_{n+1} \left(\cos\alpha\right) \right],
                & {n>0} \newline
            \displaystyle  (1 - \cos\alpha)/2,  & {n=0}
        \end{cases}

    where :math:`\alpha = \arcsin \left(\frac{r_c}{r_s} \right)` is the
    aperture angle.

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

    References
    ----------
    .. [11]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [12]  B. Rafaely and D. Khaykin, “Optimal Model-Based Beamforming and
             Independent Steering for Spherical Loudspeaker Arrays,” IEEE
             Transactions on Audio, Speech, and Language Processing, vol. 19,
             no. 7, pp. 2234-2238, 2011

    Notes
    -----
    Eq. (3) in Ref. [12]_ contains an error, here, the power of 2 on pi is
    omitted on the normalization term.

    """
    angle_cap = np.arcsin(rad_cap / rad_sphere)
    arg = np.cos(angle_cap)
    n_sh = (n_max+1)**2

    aperture = np.zeros((n_sh, n_sh), dtype=float)

    aperture[0, 0] = (1-arg)*2*np.pi
    for n in range(1, n_max+1):
        legendre_minus = special.legendre(n-1)(arg)
        legendre_plus = special.legendre(n+1)(arg)
        legendre_term = legendre_minus - legendre_plus
        for m in range(-n, n+1):
            acn = nm2acn(n, m)
            aperture[acn, acn] = legendre_term * 4 * np.pi / (2*n+1)

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
    source. [13]_, [14]_

    TODO: This function does not have a test yet.

    References
    ----------
    .. [13]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [14]  F. Zotter, A. Sontacchi, and R. Höldrich, “Modeling a spherical
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
    radiation = np.zeros((n_bins, n_sh, n_sh), dtype=complex)

    for n in range(n_max+1):
        hankel = _special.spherical_hankel(n, k*distance, kind=2)
        hankel_prime = _special.spherical_hankel(
            n, k*rad_sphere, kind=2, derivative=True)
        radiation_order = -1j * hankel/hankel_prime * \
            density_medium * speed_of_sound
        for m in range(-n, n+1):
            acn = nm2acn(n, m)
            radiation[:, acn, acn] = radiation_order

    return radiation


def sid(n_max):
    """Calculates the SID indices up to spherical harmonic order n_max.
    The SID indices were originally proposed by Daniel [#]_, more recently
    ACN indexing has been favored and is used in the AmbiX format [#]_.

    Parameters
    ----------
    n_max : int
        The maximum spherical harmonic order

    Returns
    -------
    sid_n : array-like, int
        The SID indices for all orders
    sid_m : array-like, int
        The SID indices for all degrees

    References
    ----------
    .. [#]  J. Daniel, “Représentation de champs acoustiques, application à la
            transmission et à la reproduction de scènes sonores complexes dans
            un contexte multimédia,” Dissertation, l’Université Paris 6, Paris,
            2001.

    .. [#]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
            Suggested Ambisonics Format (revised by F. Zotter),” International
            Symposium on Ambisonics and Spherical Acoustics,
            vol. 3, pp. 1-11, 2011.

    """
    n_sh = (n_max+1)**2
    sid_n = sph_identity_matrix(n_max, 'n-nm').T @ np.arange(0, n_max+1)
    sid_m = np.zeros(n_sh, dtype=int)
    idx_n = 0
    for n in range(1, n_max+1):
        for m in range(1, n+1):
            sid_m[idx_n + 2*m-1] = n-m+1
            sid_m[idx_n + 2*m] = -(n-m+1)
        sid_m[idx_n + 2*n + 1] = 0
        idx_n += 2*n+1

    return sid_n, sid_m


def sid2acn(n_max):
    """Convert from SID channel indexing to ACN indeces.
    Returns the indices to achieve a corresponding linear acn indexing.

    Parameters
    ----------
    n_max : int
        The maximum spherical harmonic order.

    Returns
    -------
    acn : array-like, int
        The SID indices sorted according to a respective linear ACN indexing.
    """
    sid_n, sid_m = sid(n_max)
    linear_sid = nm2acn(sid_n, sid_m)
    return np.argsort(linear_sid)


def sph_identity_matrix(n_max, type='n-nm'):
    """Calculate a spherical harmonic identity matrix.

    Parameters
    ----------
    n_max : int
        The spherical harmonic order.
    type : str, optional
        The type of identity matrix. Currently only 'n-nm' is implemented.

    Returns
    -------
    identity_matrix : array-like, int
        The spherical harmonic identity matrix.

    Examples
    --------

    The identity matrix can for example be used to decompress from order only
    vectors to a full order and degree representation.

    >>> import spharpy
    >>> import matplotlib.pyplot as plt
    >>> n_max = 2
    >>> E = spharpy.spherical.sph_identity_matrix(n_max, type='n-nm')
    >>> a_n = [1, 2, 3]
    >>> a_nm = E.T @ a_n
    >>> a_nm
    array([1, 2, 2, 2, 3, 3, 3, 3, 3])

    The matrix E in this case has the following form.

    .. plot::

        >>> import spharpy
        >>> import matplotlib.pyplot as plt
        >>> n_max = 2
        >>> E = spharpy.spherical.sph_identity_matrix(n_max, type='n-nm')
        >>> plt.matshow(E, cmap=plt.get_cmap('Greys'))
        >>> plt.gca().set_aspect('equal')

    """
    n_sh = (n_max+1)**2

    if type != 'n-nm':
        raise NotImplementedError

    identity_matrix = np.zeros((n_max+1, n_sh), dtype=int)

    for n in range(n_max+1):
        m = np.arange(-n, n+1)
        linear_nm = nm2acn(np.tile(n, m.shape), m)
        identity_matrix[n, linear_nm] = 1

    return identity_matrix
