import numpy as np
import numpy.polynomial as poly
from scipy.linalg import eig
from scipy.special import factorial

import spharpy
import spharpy.special as special


def dolph_chebyshev_weights(
        n_max,
        design_parameter,
        design_criterion='sidelobe'):
    """Calculate the weights for a spherical Dolph-Chebyshev beamformer. The
    design criterion can either be a desired side-lobe attenuation or a desired
    main-lobe width. Once one criterion is chosen, the other will become a
    dependent property which will be chosen accordingly.

    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    design_parameter : float, double
        This can either be the desired side-lobe attenuation or the width of
        the main-lobe in radians.
    design_criterion : 'sidelobe', 'mainlobe'
        Whether the design parameter argument is the desired side-lobe
        attenuation or the desired main-lobe width.

    Returns
    -------
    weigths : ndarray, double
        An array containing the weight coefficients $d_nm$.

    References
    ----------
    ..  [1] A. Koretz and B. Rafaely, “Dolph-Chebyshev beampattern design for
        spherical arrays,” IEEE Transactions on Signal Processing, vol. 57,
        no. 6, pp. 2417–2420, 2009.

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
                    temp = temp+(1-(-1)**(m+i+1))/(m+i+1) * \
                           factorial(j)/(factorial(m)*factorial(j-m)) * \
                           (1/2**j)*t_2N[2*j]*P_N[i, n]*x0**(2*j)
        d_n[n] = (2*np.pi/R)*temp

    weights = spharpy.indexing.sph_identity_matrix(n_max, type='n-nm').T @ d_n

    return weights


def rE_max_weights(n_max):
    """Weights that maximize the length of the energy vector.
    This is most often used in Ambisonics decoding.

    Parameters
    ----------
    n_max : int
        Spherical harmonic order

    Returns
    -------
    weights : ndarray, double
        An array containing the weight coefficients.

    References
    ----------
    ..  [2] J. Daniel, J.-B. Rault, and J.-D. Polack, “Ambisonics Encoding of
        Other Audio Formats for Multiple Listening Conditions,” in 105th
        Convention of the Audio Engineering Society, 1998, vol. 3.

    """
    leg = poly.legendre.Legendre.basis(n_max+1)
    P_n_root = poly.legendre.legroots(leg.coef)
    max_root = np.max(np.abs(P_n_root))
    g_n = np.zeros(n_max+1)
    for n in range(0, n_max+1):
        leg = poly.legendre.Legendre.basis(n)
        g_n[n] = leg(max_root)

    weights = spharpy.indexing.sph_identity_matrix(n_max).T @ g_n

    return weights


def maximum_front_back_ratio_weights(n_max):
    """Weights that maximize the front-back ratio of the beam pattern.
    This is also often referred to as the super-cardioid beam pattern.

    Parameters
    ----------
    n_max : int
        The spherical harmonic order

    Returns
    -------
    weigths : ndarray, double
        An array containing the weight coefficients

    Note
    ----
    The weights are calculated from an eigenvalue problem

    References
    ----------
    [3] B. Rafaely, Fundamentals of Spherical Array Processing, Springer, 2015.

    """
    P_N = np.zeros((n_max+1, n_max+1))
    for n in range(n_max+1):
        P_N[0:n+1, n] = special.legendre_coefficients(n)

    Ann = np.zeros((n_max+1, n_max+1))
    Bnn = np.zeros((n_max+1, n_max+1))
    for n in range(n_max+1):
        for n_dash in range(n_max+1):
            const = 1/8/np.pi * (2*n+1) * (2*n_dash+1)
            temp = 0
            for q in range(0, n+1):
                for ll in range(0, n_dash+1):
                    temp += 1/(q+ll+1) * P_N[q, n] * P_N[ll, n_dash]
            Ann[n, n_dash] = temp * const

            temp = 0
            for q in range(0, n+1):
                for ll in range(0, n_dash+1):
                    temp += ((-1)**(q+ll))/(q+ll+1) * \
                        P_N[q, n] * P_N[ll, n_dash]
            Bnn[n, n_dash] = temp * const

    eigenvals, eigenvectors = eig(Ann, Bnn)
    f_n = eigenvectors[:, np.argmax(np.real(eigenvals))]
    weights = spharpy.indexing.sph_identity_matrix(n_max).T @ f_n

    return weights
