"""Beamforming methods for spherical harmonic signals."""

import itertools
import numpy as np
import numpy.polynomial as poly
from scipy.linalg import eigh
from scipy.special import factorial

import spharpy.special as special
from spharpy.spherical import sph_identity_matrix


def dolph_chebyshev_weights(
        n_max,
        design_parameter,
        design_criterion='sidelobe'):
    r"""
    Calculate the weights for a spherical Dolph-Chebyshev beamformer.

    The design criterion can either be a desired side-lobe attenuation or a
    desired main-lobe width. Once one criterion is chosen, the other will
    become a dependent property which will be chosen accordingly [#]_.

    Parameters
    ----------
    n_max : int
        Spherical harmonic order. Must be an integer greater than zero.
    design_parameter : float
        This can either be the desired side-lobe attenuation or the width of
        the main-lobe in radians.
    design_criterion : str
        Can be either ``'sidelobe'`` or ``'mainlobe'`` (see
        `design_parameter` above). The default is ``'sidelobe'``.

    Returns
    -------
    weights : array-like, float
        A flat array containing the :math:`(n_\mathrm{max} + 1)^2` beamformer
        weights.

    References
    ----------
    .. [#]  A. Koretz and B. Rafaely, “Dolph-Chebyshev beampattern design for
            spherical arrays,” IEEE Transactions on Signal Processing, vol. 57,
            no. 6, pp. 2417-2420, 2009.

    Examples
    --------
    Apply Dolph-Chebyshev beamformers with a side-lobe
    attenuation of 50 dB to a plane wave from zero degrees azimuth.

    .. plot::

        >>> import spharpy
        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>>
        >>> # use real valued spherical harmonics of order 7
        >>> definition = spharpy.SphericalHarmonicDefinition(n_max=7)
        >>>
        >>> # define the sound field (plane wave from 0 degree azimuth)
        >>> soundfield = spharpy.SphericalHarmonics.from_definition(
        ...     definition, pf.Coordinates(1, 0, 0))
        >>> a_nm = soundfield.basis
        >>>
        >>> # beamforming
        >>> # a) generate 500 steering vectors
        >>> steering = spharpy.SphericalHarmonics.from_definition(
        ...     definition, pf.Coordinates.from_spherical_elevation(
        ...         np.linspace(0, 2*np.pi, 500), 0, 1))
        >>> Y_steering = steering.basis
        >>>
        >>> # b) design beamformer weights
        >>> R = 10**(50/20)
        >>> d_nm = spharpy.beamforming.dolph_chebyshev_weights(
        ...     definition.n_max, R, design_criterion='sidelobe')
        >>>
        >>> # c) appling beamformer weights yields 500 beamformers
        >>> beamformer = np.squeeze(Y_steering @ np.diag(d_nm))
        >>>
        >>> # d) apply beamformers to the soundfield
        >>> soundfield_beamformed = beamformer @ a_nm.T
        >>>
        >>> # plot soundfield evaluated with beamformers
        >>> ax = plt.axes(projection='polar')
        >>> ax.plot(steering.coordinates.azimuth,
        ...         20*np.log10(np.abs(soundfield_beamformed)))
        >>> ax.set_rticks([-50, -25, 0])
        >>> ax.set_theta_zero_location('N')
        >>> ax.set_xlabel('Azimuth in degree')
    """
    M = 2*n_max
    if design_criterion == 'sidelobe':
        R = design_parameter
        x0 = np.cosh((1/M) * np.arccosh(R))
    elif design_criterion == 'mainlobe':
        theta0 = design_parameter
        x0 = np.cos(np.pi/2/M) / np.cos(theta0/2)
        R = np.cosh(M * np.arccosh(x0))
    else:
        raise ValueError("This design criterion is not available.")

    t_2N = special.chebyshev_coefficients(2*n_max)

    P_N = np.zeros((n_max+1, n_max+1))
    for n in range(n_max+1):
        P_N[0:n+1, n] = special.legendre_coefficients(n)

    d_n = np.zeros(n_max+1)
    for n in range(n_max+1):
        temp = 0
        for i in range(n+1):
            for j in range(n_max+1):
                for m in range(j+1):
                    temp += (1-(-1)**(m+i+1))/(m+i+1) * \
                        factorial(j)/(factorial(m)*factorial(j-m)) * \
                        (1/2**j)*t_2N[2*j]*P_N[i, n]*x0**(2*j)
        d_n[n] = (2*np.pi/R)*temp

    return sph_identity_matrix(n_max, matrix_type='n-nm').T @ d_n


def rE_max_weights(n_max, normalize=True):
    r"""
    Calculate weights for a max :math:`\mathrm{r}_\mathrm{E}` beamformer.

    The weights that maximize the length of the energy vector.
    This is most often used in Ambisonics decoding [#]_.

    Parameters
    ----------
    n_max : int
        Spherical harmonic order. Must be an integer greater than zero.
    normalize : bool
        If ``True``, the weights will be normalized such that the complex
        amplitude of a plane wave is not distorted. The default is ``True``.

    Returns
    -------
    weights : array-like, float
        A flat array containing the :math:`(n_\mathrm{max} + 1)^2` beamformer
        weights.

    References
    ----------
    .. [#]  J. Daniel, J.-B. Rault, and J.-D. Polack, “Ambisonics Encoding of
            Other Audio Formats for Multiple Listening Conditions,” in 105th
            Convention of the Audio Engineering Society, 1998, vol. 3.

    Examples
    --------
    Apply max :math:`\mathrm{r}_\mathrm{E}` beamformers to a plane wave from
    zero degrees azimuth.

    .. plot::

        >>> import spharpy
        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>>
        >>> # use real valued spherical harmonics of order 7
        >>> definition = spharpy.SphericalHarmonicDefinition(n_max=7)
        >>>
        >>> # define the sound field (plane wave from 0 degree azimuth)
        >>> soundfield = spharpy.SphericalHarmonics.from_definition(
        ...     definition, pf.Coordinates(1, 0, 0))
        >>> a_nm = soundfield.basis
        >>>
        >>> # beamforming
        >>> # a) generate 500 steering vectors
        >>> steering = spharpy.SphericalHarmonics.from_definition(
        ...     definition, pf.Coordinates.from_spherical_elevation(
        ...         np.linspace(0, 2*np.pi, 500), 0, 1))
        >>> Y_steering = steering.basis
        >>>
        >>> # b) design beamformer weights
        >>> d_nm = spharpy.beamforming.rE_max_weights(definition.n_max)
        >>>
        >>> # c) appling beamformer weights yields 500 beamformers
        >>> beamformer = np.squeeze(Y_steering @ np.diag(d_nm))
        >>>
        >>> # d) apply beamformers to the soundfield
        >>> soundfield_beamformed = beamformer @ a_nm.T
        >>>
        >>> # plot soundfield evaluated with beamformers
        >>> ax = plt.axes(projection='polar')
        >>> ax.plot(steering.coordinates.azimuth,
        ...         20*np.log10(np.abs(soundfield_beamformed)))
        >>> ax.set_rticks([-50, -25, 0])
        >>> ax.set_theta_zero_location('N')
        >>> ax.set_xlabel('Azimuth in degree')
    """
    leg = poly.legendre.Legendre.basis(n_max+1)
    P_n_root = poly.legendre.legroots(leg.coef)
    max_root = np.max(np.abs(P_n_root))
    g_n = np.zeros(n_max+1)
    for n in range(n_max+1):
        leg = poly.legendre.Legendre.basis(n)
        g_n[n] = leg(max_root)

    if normalize:
        g_n = normalize_beamforming_weights(g_n, n_max)

    return sph_identity_matrix(n_max).T @ g_n


def maximum_front_back_ratio_weights(n_max, normalize=True):
    r"""
    Compute weights that maximize the front-back ratio of the beam pattern.

    This is also often referred to as the super-cardioid beam pattern.
    The weights are calculated from an eigenvalue problem [#]_. For high
    spherical harmonic orders, the eigenvalue problem may not be feasible and
    a solution will not be found.

    Parameters
    ----------
    n_max : int
        The spherical harmonic order. Must be an integer greater than zero.
    normalize : bool
        If ``True``, the weights will be normalized such that the complex
        amplitude of a plane wave is not distorted. The default is ``True``.

    Returns
    -------
    weights : array-like, float
        A flat array containing the :math:`(n_\mathrm{max} + 1)^2` beamformer
        weights.

    References
    ----------
    .. [#]  B. Rafaely, Fundamentals of Spherical Array Processing,
            Springer, 2015.

    Examples
    --------
    Apply beamformers maximizing the front-back ratio to a plane wave from
    zero degrees azimuth.

    .. plot::

        >>> import spharpy
        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>>
        >>> # use real valued spherical harmonics of order 7
        >>> definition = spharpy.SphericalHarmonicDefinition(n_max=7)
        >>>
        >>> # define the sound field (plane wave from 0 degree azimuth)
        >>> soundfield = spharpy.SphericalHarmonics.from_definition(
        ...     definition, pf.Coordinates(1, 0, 0))
        >>> a_nm = soundfield.basis
        >>>
        >>> # beamforming
        >>> # a) generate 500 steering vectors
        >>> steering = spharpy.SphericalHarmonics.from_definition(
        ...     definition, pf.Coordinates.from_spherical_elevation(
        ...         np.linspace(0, 2*np.pi, 500), 0, 1))
        >>> Y_steering = steering.basis
        >>>
        >>> # b) design beamformer weights
        >>> R = 10**(50/20)
        >>> d_nm = spharpy.beamforming.maximum_front_back_ratio_weights(
        ...     definition.n_max)
        >>>
        >>> # c) appling beamformer weights yields 500 beamformers
        >>> beamformer = np.squeeze(Y_steering @ np.diag(d_nm))
        >>>
        >>> # d) apply beamformers to the soundfield
        >>> soundfield_beamformed = beamformer @ a_nm.T
        >>>
        >>> # plot soundfield evaluated with beamformers
        >>> ax = plt.axes(projection='polar')
        >>> ax.plot(steering.coordinates.azimuth,
        ...         20*np.log10(np.abs(soundfield_beamformed)))
        >>> ax.set_rticks([-50, -25, 0])
        >>> ax.set_theta_zero_location('N')
        >>> ax.set_xlabel('Azimuth in degree')
    """
    P_N = np.zeros((n_max+1, n_max+1))
    for n in range(n_max+1):
        P_N[0:n+1, n] = special.legendre_coefficients(n)

    Ann = np.zeros((n_max+1, n_max+1))
    Bnn = np.zeros((n_max+1, n_max+1))
    for n in range(n_max+1):
        for n_dash in range(n_max+1):
            const = 1/8/np.pi * (2*n+1) * (2*n_dash+1)
            temp = sum(
                1 / (q+ll+1) * P_N[q, n] * P_N[ll, n_dash]
                for q, ll in itertools.product(range(n+1), range(n_dash+1)))
            Ann[n, n_dash] = temp * const

            temp = sum(
                ((-1) ** (q+ll)) / (q+ll+1) * P_N[q, n] * P_N[ll, n_dash]
                for q, ll in itertools.product(range(n+1), range(n_dash+1)))
            Bnn[n, n_dash] = temp * const

    try:
        eigenvals, eigenvectors = eigh(Ann, Bnn)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            'Eigenvalue decomposition did not converge. '
            'Try reducing the spherical harmonic order.') from e
    f_n = eigenvectors[:, np.argmax(np.real(eigenvals))]
    if normalize:
        f_n = normalize_beamforming_weights(f_n, n_max)
    else:
        f_n /= np.sign(f_n[0])

    return sph_identity_matrix(n_max).T @ f_n


def normalize_beamforming_weights(weights, n_max):
    r"""
    Normalize the beamforming weights.

    The weights are normalized such that the complex amplitude of a
    plane wave is not distorted.

    Parameters
    ----------
    weights : array-like, float
        An array containing the beamforming weights. The array must be of
        shaoe :math:`(\dots, (n_\mathrm{max}+1)^2)`.
    n_max : int
        The spherical harmonic order. Must be an integer greater than zero.

    Returns
    -------
    weights : array-like, float
        An array containing the normalized beamforming weights

    Examples
    --------
    Calculate and normalize hann window function based beamforming weights.

    >>> import spharpy
    >>> from scipy.signal.windows import hann
    ...
    >>> tapering_window = hann(2*(N+1)+1)[N+1:-1], N)
    >>> h_n = spharpy.beamforming.normalize_beamforming_weights(
    ...     tapering_window, N)

    """
    return weights / np.dot(weights, 2*np.arange(0, n_max+1)+1) * (4*np.pi)
