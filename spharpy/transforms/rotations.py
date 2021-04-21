"""
Rotation/Translation operations for data in the spherical harmonic domains
"""

import numpy as np
import spharpy
from scipy.special import eval_jacobi, factorial
from scipy.spatial.transform import Rotation


class RotationSH(Rotation):
    """Class for rotations of coordinates and data.

    """

    def __init__(self, quat, n_max=0, *args, **kwargs):
        """Initialize

        Parameters
        ----------
        quat : array_like, shape (N, 4) or (4,)
            Each row is a (possibly non-unit norm) quaternion in scalar-last
            (x, y, z, w) format. Each quaternion will be normalized to unit
            norm.
        n_max : int
            The spherical harmonic order

        Returns
        -------
        RotationSH
            The rotation object with spherical harmonic order n_max.

        Note
        ----
        Initializing using the constructor is not advised. Always use the
        respective ``from_quat`` method.

        """
        super().__init__(quat, *args, **kwargs)
        if n_max < 0:
            raise ValueError("The order needs to be a positive value.")
        self._n_max = int(n_max)

    @classmethod
    def from_rotvec(cls, n_max, rotvec, degrees=False, *args, **kwargs):
        """Initialize from rotation vectors.
        A rotation vector is a 3 dimensional vector which is co-directional to
        the axis of rotation and whose norm gives the angle of rotation [#]_.

        Parameters
        ----------
        n_max : int
            Spherical harmonic order
        rotvec : array_like, float, shape (L, 3) or (3,)
            A single vector or a stack of vectors, where rot_vec[i] gives the
            ith rotation vector.
        degrees : bool, optional
            Specify if rotation angles are defined in degrees instead of
            radians, by default False.

        Returns
        -------
        RotationSH
            Object containing the rotations represented by input rotation
            vectors.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation\
#Rotation_vector

        Examples
        --------
        >>> from spharpy.transforms import Rotation as R
        >>> rot = R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))
        >>> rot.as_spherical_harmonic()

        """
        if degrees:
            rotvec = np.deg2rad(rotvec)

        cls = super(RotationSH, cls).from_rotvec(rotvec, *args, **kwargs)
        cls.n_max = n_max
        return cls

    @classmethod
    def from_euler(cls, n_max, seq, angles, degrees=False, **kwargs):
        """Initialize from Euler angles.

        Rotations in 3-D can be represented by a sequence of 3 rotations
        around a sequence of axes. In theory, any three axes spanning the 3-D
        Euclidean space are enough. In practice, the axes of rotation are
        chosen to be the basis vectors.
        The three rotations can either be in a global frame of reference
        (extrinsic) or in a body centred frame of reference (intrinsic), which
        is attached to, and moves with, the object under rotation [#]_.

        Parameters
        ----------
        n_max : int
            Spherical harmonic order.
        seq : str
            Specifies sequence of axes for rotations. Up to 3 characters
            belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic rotations,
            or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and intrinsic
            rotations cannot be mixed in one function call.

        angles : (float or array_like, shape (N,) or (N, [1 or 2 or 3]))
            Euler angles specified in radians (degrees is False) or degrees
            (degrees is True). For a single character seq, angles can be:

            - a single value
            - array_like with shape (N,), where each ``angle[i]`` corresponds
              to a single rotation
            - array_like with shape (N, 1), where each ``angle[i, 0]``
              corresponds to a single rotation

            For 2- and 3-character wide seq, angles can be:

            - array_like with shape (W,) where W is the width of ``seq``, which
              corresponds to a single rotation with W axes
            - array_like with shape (N, W) where each ``angle[i]`` corresponds
              to a sequence of Euler angles describing a single rotation

        degrees : bool, optional
            If True, then the given angles are assumed to be in degrees.
            Default is False.

        Returns
        -------
        RotationSH
            Object containing the rotation represented by the sequence of
            rotations around given axes with given angles.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Euler_angles

        Examples
        --------
        >>> from spharpy.transforms import Rotation as R
        >>> rot = R.from_euler('z', 90, degrees=True)
        >>> rot.as_spherical_harmonic()


        """
        cls = super(RotationSH, cls).from_euler(
            seq, angles, degrees=degrees, **kwargs)
        cls.n_max = n_max
        return cls

    @classmethod
    def from_quat(cls, n_max, quat, **kwargs):
        """Initialize from quaternions.
        3D rotations can be represented using unit-norm quaternions [#]_.

        Parameters
        ----------
        n_max : int
            Spherical harmonic order
        quat : (array_like, shape (N, 4) or (4,))
            Each row is a (possibly non-unit norm) quaternion in scalar-last
            (x, y, z, w) format. Each quaternion will be normalized to unit
            norm.

        Returns
        -------
        RotationSH
            Object containing the rotations represented by input quaternions.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

        Examples
        --------
        >>> from spharpy.transforms import Rotation as R
        >>> rot = R.from_quat([1, 0, 0, 0])
        >>> rot.as_spherical_harmonic()

        """
        cls = super(RotationSH, cls).from_quat(quat, **kwargs)
        cls.n_max = n_max
        return cls

    @classmethod
    def from_matrix(cls, n_max, matrix, **kwargs):
        """Initialize from rotation matrix.
        Rotations in 3 dimensions can be represented with 3 x 3 proper
        orthogonal matrices [1]_. If the input is not proper orthogonal,
        an approximation is created using the method described in [2]_.

        Parameters
        ----------
        n_max : int
            Spherical harmonic order
        matrix : (array_like, shape (N, 3, 3) or (3, 3))
            A single matrix or a stack of matrices, where ``matrix[i]`` is
            the i-th matrix.

        Returns
        -------
        RotationSH
            Object containing the rotations represented by the rotation
            matrices.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Rotation_matrix
        .. [2] F. Landis Markley, “Unit Quaternion from Rotation Matrix”,
               Journal of guidance, control, and dynamics vol. 31.2,
               pp. 440-442, 2008.

        Examples
        --------
        >>> from spharpy.transforms import Rotation as R
        >>> rot = R.from_matrix([
        ...     [0, -1, 0],
        ...     [1,  0, 0],
        ...     [0,  0, 1]])
        >>> rot.as_spherical_harmonic()
        """
        cls = super(RotationSH, cls).from_matrix(matrix, **kwargs)
        cls.n_max = n_max
        return cls

    @property
    def n_max(self):
        """The spherical harmonic order used for spherical harmonic rotation
        matrices.
        """
        return self._n_max

    @n_max.setter
    def n_max(self, value):
        """Set the spherical harmonic order

        Parameters
        ----------
        value : int
            The spherical harmonic order used for spherical harmonic rotation
        matrices.
        """
        if value < 0:
            raise ValueError("The order needs to be a positive value.")
        self._n_max = value

    def as_spherical_harmonic(self, type='real'):
        """Export the rotation operations as a spherical harmonic rotation
        matrices. Supports complex and real-valued spherical harmonics.

        Parameters
        ----------
        real : string, optional
            Spherical harmonic definition. Can either be 'complex' or 'real',
            by default 'real' is used.

        Returns
        -------
        array, complex or float
            Stack of block-diagonal rotation matrices.
        """
        euler_angles = np.atleast_2d(self.as_euler('zyz'))
        n_matrices = euler_angles.shape[0]

        n_sh = (self.n_max+1)**2
        if type == 'real':
            dtype = np.double
            rot_func = wigner_d_rotation_real
        elif type == 'complex':
            dtype = complex
            rot_func = wigner_d_rotation
        else:
            raise ValueError("Invalid spherical harmonic type {}".format(type))

        D = np.zeros((n_matrices, n_sh, n_sh), dtype=dtype)

        for idx, angles in enumerate(euler_angles):
            D[idx, :, :] = rot_func(
                self.n_max, angles[0], angles[1], angles[2])

        return np.squeeze(D)

    def apply(self, coefficients, type='real'):
        """Apply the rotation to L sets of spherical harmonic coefficients

        Parameters
        ----------
        coefficients : array, complex, shape :math:`((n_max+1)^2, L)`
            L sets of spherical harmonic coefficients with a respective order
            :math:`((n_max+1)^2`

        Returns
        -------
        array, complex
            The rotated data
        """
        D = self.as_spherical_harmonic(type=type)
        if D.ndim > 2:
            M = np.diag(np.ones((self.n_max+1)**2))
            for d in D:
                M = M @ d
        else:
            M = D

        return M @ coefficients


def rotation_z_axis(n_max, angle):
    """Rotation matrix for complex spherical harmonics around the z-axis
    by a given angle. The rotation is performed such that positive angles
    result in a counter clockwise rotation of the data [#]_.

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
    array_like, complex, shape :math:`((n_{max}+1)^2, (n_{max}+1)^2)`
        Diagonal rotation matrix evaluated for the specified angle

    References
    ----------
    .. [#]  N. A. Gumerov and R. Duraiswami, “Recursions for the computation
            of multipole translation and rotation coefficients for the 3-d
            helmholtz equation,” vol. 25, no. 4, pp. 1344–1381, 2003.


    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 1, 0, 0])
    >>> rotMat = spharpy.transforms.rotation_z_axis(n_max, np.pi/2)
    >>> sh_vec_rotated = rotMat @ sh_vec
    """

    acn = np.arange(0, (n_max+1)**2)
    m = spharpy.spherical.acn2nm(acn)[1]
    rotation_phi = np.exp(-1j*angle*m)

    return np.diag(rotation_phi)


def rotation_z_axis_real(n_max, angle):
    """Rotation matrix for real-valued spherical harmonics around the z-axis
    by a given angle. The rotation is performed such that positive angles
    result in a counter clockwise rotation of the data [#]_.

    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    angle : number
        Rotation angle in radians `[0, 2 \\pi]`

    Returns
    -------
    array_like, float, shape :math:`((n_max+1)^2, (n_max+1)^2)`
        Block-diagonal Rotation matrix evaluated for the specified angle.

    References
    ----------
    .. [#] M. Kronlachner, “Spatial Transformations for the Alteration of
           Ambisonic Recordings,” Master Thesis, University of Music and
           Performing Arts, Graz, 2014.


    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 1, 0, 0])
    >>> rotMat = spharpy.transforms.rotation_z_axis_real(n_max, np.pi/2)
    >>> sh_vec_rotated = rotMat @ sh_vec

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
    r"""Wigner-D rotation matrix for Euler rotations by angles
    (\alpha, \beta, \gamma) around the (z,y,z)-axes.
    The implementation follows [#]_. and rotation is performed such that
    positive angles result in a counter clockwise rotation of the data.

    .. math::

        D_{m^\prime,m}^n(\alpha, \beta, \gamma) =
        e^{-im^\prime\alpha} d_{m^\prime,m}^n(\beta) e^{-im\gamma}

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
    array_like, complex,:math:`((n_max+1)^2, (n_max+1)^2)`
        Block diagonal rotation matrix


    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 0, 1, 0])
    >>> rotMat = spharpy.transforms.wigner_d_rotation(n_max, 0, np.pi/4, 0)
    >>> sh_vec_rotated = rotMat @ sh_vec

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


def wigner_d_rotation_real(n_max, alpha, beta, gamma):
    r"""Wigner-D rotation matrix for Euler rotations for real-valued spherical
    harmonics by angles (\alpha, \beta, \gamma) around the (z,y,z)-axes.
    The implementation follows [#]_ and the rotation is performed such that
    positive angles result in a counter clockwise rotation of the data.

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
    array_like, float, :math:`((n_max+1)^2, (n_max+1)^2)`
        Block diagonal rotation matrix

    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 0, 1, 0])
    >>> rotMat = spharpy.transforms.wigner_d_rotation_real(
    >>>     n_max, 0, np.pi/4, 0)
    >>> sh_vec_rotated = rotMat @ sh_vec

    References
    ----------
    .. [#]  M. A. Blanco, M. Flórez, and M. Bermejo, “Evaluation of the
            rotation matrices in the basis of real spherical harmonics,”
            Journal of Molecular Structure: THEOCHEM,  vol. 419, no. 1–3,
            pp. 19–27, Dec. 1997, doi: 10.1016/S0166-1280(97)00185-1.

    """
    n_sh = (n_max+1)**2
    R = np.zeros((n_sh, n_sh), dtype=np.double)

    for row_acn in np.arange(0, (n_max+1)**2):
        for col_acn in np.arange(0, (n_max+1)**2):
            n, m = spharpy.spherical.acn2nm(col_acn)
            n_dash, m_dash = spharpy.spherical.acn2nm(row_acn)
            if n == n_dash:
                # minus beta opposite rotation direction
                d_l_1 = wigner_d_function(n, np.abs(m_dash), np.abs(m), -beta)
                d_l_2 = wigner_d_function(n, np.abs(m), -np.abs(m_dash), -beta)

                R[row_acn, col_acn] = \
                    _sign(m_dash) * _Phi(m, alpha) * _Phi(m_dash, gamma) * \
                    (d_l_1 + (-1)**int(m) * d_l_2)/2 \
                    - _sign(m) * _Phi(-m, alpha) * _Phi(-m_dash, gamma) * \
                    (d_l_1 - (-1)**int(m) * d_l_2)/2

    return R


def _sign(x):
    """
    Returns sign of x, differs from numpy definition for x=0
    """
    if x < 0:
        sign = -1
    else:
        sign = 1

    return sign


def _Phi(m, angle):
    """
    Rotation Matrix around z-axis for real Spherical Harmonics as defined in
    Blanco et al., Evaluation of the rotation matrices in the basis of real
    spherical harmonics, eq.(8)
    """
    if m > 0:
        phi = np.sqrt(2)*np.cos(m*angle)
    elif m == 0:
        phi = 1
    elif m < 0:
        # minus due to differing phase convention
        phi = -np.sqrt(2)*np.sin(np.abs(m)*angle)

    return phi


def wigner_d_function(n, m_dash, m, beta):
    r"""Wigner-d function for rotations around the y-axis.
    Convention as defined in [#]_.

    Parameters
    ----------
    n : int
        order
    m_dash : int
        degree
    m : int
        degree
    beta : float
        Rotation angle

    Returns
    -------
    float
        Wigner-d symbol

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
