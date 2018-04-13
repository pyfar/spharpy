"""
Rotation/Translation operations for data in the spherical harmonic domains
"""

import numpy as np
import spharpy


def rotation_z_axis(n_max, angle):
    """Rotation of spherical harmonic coeffiecients around the z-axis
    by a given angle. The rotation is performed such that positive angles result
    in a counter clockwise rotation of the data [1]_.

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
    roration_matrix : ndarray
        Rotation matrix evaluated for the specified angle

    References
    ----------
    .. [1]  N. A. Gumerov and R. Duraiswami, “Recursions for the computation of multipole
            translation and rotation coefficients for the 3-d helmholtz equation,”
            vol. 25, no. 4, pp. 1344–1381, 2003.



    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 1, 0, 0])
    >>> Y_nm = spharpy.spherical.spherical_harmonic_basis(n_max, theta, phi)
    >>> rotMat = spharpy.transforms.rotation_z_axis(n_max, np.pi/2)
    >>> sh_vec_rotated =  rotMat @ sh_vec
    """

    acn = np.arange(0, (n_max+1)**2)
    n, m = spharpy.spherical.acn2nm(acn)
    rotation_phi = np.exp(-1j*angle*m)

    return np.diag(rotation_phi)
