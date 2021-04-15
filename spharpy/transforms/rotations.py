"""
Rotation/Translation operations for data in the spherical harmonic domains
"""

import numpy as np
import spharpy
from scipy.special import eval_jacobi, factorial
from scipy.spatial.transform import Rotation



# class RotationSH(Rotation):

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#             return self

#     def as_spherical_harmonic(real=True):



def rotation_z_axis(n_max, angle):
    """Rotation of spherical harmonic coeffiecients around the z-axis
    by a given angle. The rotation is performed such that positive angles
    result in a counter clockwise rotation of the data [1]_.

    .. math::

        c_{nm}(\\theta, \\phi + \\xi) = e^{-im\\xi} c_{nm}(\\theta, \\phi)

    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    angle : number
        Rotation angle in radians `[0, 2 \\pi]`

    Returns
    -------
    rotation_matrix : ndarray
        Rotation matrix evaluated for the specified angle

    References
    ----------
    .. [1]  N. A. Gumerov and R. Duraiswami, “Recursions for the computation
            of multipole translation and rotation coefficients for the 3-d
            helmholtz equation,” vol. 25, no. 4, pp. 1344–1381, 2003.



    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 1, 0, 0])
    >>> Y_nm = spharpy.spherical.spherical_harmonic_basis(n_max, theta, phi)
    >>> rotMat = spharpy.transforms.rotation_z_axis(n_max, np.pi/2)
    >>> sh_vec_rotated = rotMat @ sh_vec
    """

    acn = np.arange(0, (n_max+1)**2)
    n, m = spharpy.spherical.acn2nm(acn)
    rotation_phi = np.exp(-1j*angle*m)

    return np.diag(rotation_phi)


def rotation_z_axis_real(n_max, angle):
    """Rotation of spherical harmonic coeffiecients around the z-axis
    by a given angle. The rotation is performed such that positive angles
    result in a counter clockwise rotation of the data [1]_.
    """
    acn = np.arange(0, (n_max + 1) ** 2)
    n, m = spharpy.spherical.acn2nm(acn)
    acn_reverse_degree = n ** 2 + n - m

    rotation_phi = np.zeros(((n_max + 1) ** 2, (n_max + 1) ** 2))
    mask = m == 0
    rotation_phi[acn[mask], acn[mask]] = 1.0

    mask_pos = m > 0
    mask_neg = m < 0
    rotation_phi[acn[mask_pos], acn[mask_pos]] = np.cos(
        np.abs(m[mask_pos]) * angle)
    rotation_phi[acn[mask_neg], acn[mask_neg]] = np.cos(
        np.abs(m[mask_neg]) * angle)

    # non diagonal
    rotation_phi[acn[mask_pos], acn_reverse_degree[mask_pos]] = -np.sin(
        np.abs(m[mask_pos]) * angle)
    rotation_phi[acn[mask_neg], acn_reverse_degree[mask_neg]] = np.sin(
        np.abs(m[mask_neg]) * angle)

    return rotation_phi


def wigner_d_rotation(n_max, alpha, beta, gamma):
    """Wigner-D rotation matrix

    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    alpha : float
        First z-axis rotation angle
    beta : float
        Y-axis rotation angle
    gamma : float
        Second z-axis rotation angle

    Returns
    -------
    array
        Block diagonal rotation matrix

    References
    ----------
    .. [#] B. Rafaely, Fundamentals of Spherical Array Processing, 1st ed.,
           vol. 8. Springer-Verlag GmbH Berlin Heidelberg, 2015.

    """

    n_sh = (n_max+1)**2

    R = np.zeros((n_sh, n_sh), dtype=complex)

    for row in np.arange(0, (n_max+1)**2):
        n_dash, m_dash = spharpy.spherical.acn2nm(row)
        for column in np.arange(0, (n_max+1)**2):
            n, m = spharpy.spherical.acn2nm(column)
            if n == n_dash:
                rot_alpha = np.exp(-1j*m_dash*alpha)
                rot_beta = wigner_d_function(n, m_dash, m, beta)
                rot_gamma = np.exp(-1j*m*gamma)
                R[row, column] = rot_alpha * rot_beta * rot_gamma

    return R


def wigner_d_function(n, m_dash, m, beta):
    """Wigner-D function

    Parameters
    ----------
    n : int
        order
    m_dash :
        degree
    m : [type]
        degree
    beta : float
        Rotation angle

    Returns
    -------
    [type]
        [description]

    Parameters
    ----------

    References
    ----------
    .. [#] B. Rafaely, Fundamentals of Spherical Array Processing, 1st ed.,
           vol. 8. Springer-Verlag GmbH Berlin Heidelberg, 2015.

    """

    if m >= m_dash:
        sign = 1
    else:
        sign = (-1)**int(m-m_dash)

    mu = np.abs(m_dash - m)
    nu = np.abs(m_dash + m)
    s = n - (mu + nu)/2

    norm = (factorial(s)*factorial(s+mu+nu))/(factorial(s+mu)*factorial(s+nu))
    P = eval_jacobi(s, mu, nu, np.cos(beta))
    d = sign * np.sqrt(norm) * np.sin(beta/2)**mu * np.cos(beta/2)**nu * P

    return d
